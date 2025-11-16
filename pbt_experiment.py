# pbt_experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import copy
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


@dataclass
class PBTConfig:
    population_size: int = 8
    exploit_interval: int = 5  # epochs between exploit/explore steps
    truncation_factor: float = 0.25  # top 25% replace bottom 25%
    # Hyperparameter search spaces
    lr_bounds: Tuple[float, float] = (1e-5, 1e-2)
    momentum_bounds: Tuple[float, float] = (0.8, 0.95)
    weight_decay_bounds: Tuple[float, float] = (1e-6, 1e-2)


class PBTMember:
    def __init__(self, model_class, model_args, hyperparams: Dict[str, float], member_id: int):
        self.member_id = member_id
        self.hyperparams = hyperparams
        self.model = model_class(**model_args)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=hyperparams['lr'],
            momentum=hyperparams['momentum'],
            weight_decay=hyperparams['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.performance_history = []
        self.current_performance = 0.0
        self.steps = 0
        
    def update_hyperparams(self, new_hyperparams: Dict[str, float]):
        """Update hyperparameters and recreate optimizer"""
        self.hyperparams = new_hyperparams
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=new_hyperparams['lr'],
            momentum=new_hyperparams['momentum'],
            weight_decay=new_hyperparams['weight_decay']
        )
    
    def save_checkpoint(self):
        """Save model and optimizer state"""
        return {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
            'hyperparams': copy.deepcopy(self.hyperparams),
            'performance': self.current_performance,
            'steps': self.steps
        }
    
    def load_checkpoint(self, checkpoint):
        """Load model and optimizer state"""
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.hyperparams = checkpoint['hyperparams']
        self.current_performance = checkpoint['performance']
        self.steps = checkpoint['steps']


class PopulationBasedTraining:
    def __init__(self, config: PBTConfig, model_class, model_args):
        self.config = config
        self.model_class = model_class
        self.model_args = model_args
        self.population: List[PBTMember] = []
        self.best_performance = 0.0
        self.best_member = None
        
    def initialize_population(self):
        """Initialize population with random hyperparameters"""
        for i in range(self.config.population_size):
            hyperparams = {
                'lr': self._sample_log_uniform(*self.config.lr_bounds),
                'momentum': random.uniform(*self.config.momentum_bounds),
                'weight_decay': self._sample_log_uniform(*self.config.weight_decay_bounds)
            }
            member = PBTMember(self.model_class, self.model_args, hyperparams, i)
            self.population.append(member)
    
    def _sample_log_uniform(self, low, high):
        """Sample from log-uniform distribution"""
        return np.exp(random.uniform(np.log(low), np.log(high)))
    
    def _perturb_hyperparam(self, value, bounds, is_log=True):
        """Perturb hyperparameter with random noise"""
        if is_log:
            low, high = np.log(bounds[0]), np.log(bounds[1])
            current = np.log(value)
        else:
            low, high = bounds
            current = value
        
        # Random walk with reflection at boundaries
        perturbed = current + random.uniform(-0.2, 0.2) * (high - low)
        perturbed = max(min(perturbed, high), low)  # Clip to bounds
        
        return np.exp(perturbed) if is_log else perturbed
    
    def exploit_and_explore(self):
        """PBT core algorithm: exploit top performers, explore new hyperparameters"""
        # Rank population by performance
        ranked_members = sorted(self.population, 
                              key=lambda x: x.current_performance, 
                              reverse=True)
        
        num_truncate = int(self.config.truncation_factor * self.config.population_size)
        top_members = ranked_members[:num_truncate]
        bottom_members = ranked_members[-num_truncate:]
        
        print(f"\n=== PBT Exploit & Explore ===")
        print(f"Top performers: {[m.current_performance for m in top_members]}")
        print(f"Bottom performers: {[m.current_performance for m in bottom_members]}")
        
        # Exploit: Copy from top to bottom
        for bottom_member in bottom_members:
            # Randomly select a top member to copy from
            donor = random.choice(top_members)
            checkpoint = donor.save_checkpoint()
            bottom_member.load_checkpoint(checkpoint)
            
            # Explore: Perturb hyperparameters
            new_hyperparams = {
                'lr': self._perturb_hyperparam(
                    bottom_member.hyperparams['lr'], 
                    self.config.lr_bounds, 
                    is_log=True
                ),
                'momentum': self._perturb_hyperparam(
                    bottom_member.hyperparams['momentum'],
                    self.config.momentum_bounds,
                    is_log=False
                ),
                'weight_decay': self._perturb_hyperparam(
                    bottom_member.hyperparams['weight_decay'],
                    self.config.weight_decay_bounds,
                    is_log=True
                )
            }
            
            bottom_member.update_hyperparams(new_hyperparams)
            
            print(f"Member {bottom_member.member_id} copied from {donor.member_id}")
            print(f"  New hyperparams: LR={new_hyperparams['lr']:.2e}, "
                  f"Momentum={new_hyperparams['momentum']:.3f}, "
                  f"WD={new_hyperparams['weight_decay']:.2e}")
    
    def train_member_one_epoch(self, member: PBTMember, train_loader, device):
        """Train one member for one epoch"""
        member.model.train()
        member.model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            member.optimizer.zero_grad()
            output = member.model(data)
            loss = criterion(output, target)
            loss.backward()
            member.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            member.steps += 1
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        member.current_performance = accuracy
        member.performance_history.append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate_member(self, member: PBTMember, test_loader, device):
        """Evaluate member on test set"""
        member.model.eval()
        member.model.to(device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = member.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        member.current_performance = accuracy
        return accuracy
    
    def get_population_stats(self):
        """Get statistics about current population"""
        performances = [m.current_performance for m in self.population]
        lrs = [m.hyperparams['lr'] for m in self.population]
        momentums = [m.hyperparams['momentum'] for m in self.population]
        weight_decays = [m.hyperparams['weight_decay'] for m in self.population]
        
        return {
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'best_performance': np.max(performances),
            'mean_lr': np.mean(lrs),
            'mean_momentum': np.mean(momentums),
            'mean_weight_decay': np.mean(weight_decays)
        }


# Simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_pbt_experiment():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    
    # PBT Configuration
    pbt_config = PBTConfig(
        population_size=4,  # Small for quick testing
        exploit_interval=2,  # Exploit every 2 epochs
        truncation_factor=0.25
    )
    
    # Initialize PBT
    pbt = PopulationBasedTraining(
        config=pbt_config,
        model_class=SimpleCNN,
        model_args={'num_classes': 10}
    )
    pbt.initialize_population()
    
    # Training loop
    num_epochs = 10  # Short run for demonstration
    history = []
    
    print("Starting PBT Training...")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # Train all members
        for i, member in enumerate(pbt.population):
            loss, acc = pbt.train_member_one_epoch(member, train_loader, device)
            print(f"Member {i}: Loss={loss:.4f}, Train Acc={acc:.2f}%")
        
        # Evaluate all members
        print("\nEvaluating population...")
        for member in pbt.population:
            test_acc = pbt.evaluate_member(member, test_loader, device)
            print(f"Member {member.member_id}: Test Acc={test_acc:.2f}%")
        
        # PBT step: exploit and explore
        if (epoch + 1) % pbt_config.exploit_interval == 0:
            pbt.exploit_and_explore()
        
        # Record statistics
        stats = pbt.get_population_stats()
        history.append(stats)
        print(f"\nPopulation Stats: Mean Acc={stats['mean_performance']:.2f}%, "
              f"Best={stats['best_performance']:.2f}%")
    
    # Final results
    best_member = max(pbt.population, key=lambda x: x.current_performance)
    print(f"\n=== PBT Training Complete ===")
    print(f"Best member: {best_member.member_id}")
    print(f"Best accuracy: {best_member.current_performance:.2f}%")
    print(f"Best hyperparameters: LR={best_member.hyperparams['lr']:.2e}, "
          f"Momentum={best_member.hyperparams['momentum']:.3f}, "
          f"WD={best_member.hyperparams['weight_decay']:.2e}")
    
    # Plot results
    plot_pbt_results(history, pbt.population)


def plot_pbt_results(history, population):
    """Plot PBT training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot population performance
    epochs = range(1, len(history) + 1)
    mean_perf = [h['mean_performance'] for h in history]
    best_perf = [h['best_performance'] for h in history]
    
    ax1.plot(epochs, mean_perf, 'b-', label='Mean Population')
    ax1.plot(epochs, best_perf, 'r-', label='Best Population')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Population Performance')
    ax1.legend()
    ax1.grid(True)
    
    # Plot individual member performance
    for member in population:
        ax2.plot(range(1, len(member.performance_history) + 1), 
                member.performance_history, 
                label=f'Member {member.member_id}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Individual Member Performance')
    ax2.legend()
    ax2.grid(True)
    
    # Plot hyperparameter evolution
    lr_history = [[] for _ in population]
    for epoch_data in history:
        for i, member in enumerate(population):
            lr_history[i].append(member.hyperparams['lr'])
    
    for i, lrs in enumerate(lr_history):
        ax3.semilogy(epochs, lrs, label=f'Member {i}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Evolution')
    ax3.legend()
    ax3.grid(True)
    
    # Final hyperparameter distribution
    final_lrs = [m.hyperparams['lr'] for m in population]
    final_wds = [m.hyperparams['weight_decay'] for m in population]
    final_perf = [m.current_performance for m in population]
    
    scatter = ax4.scatter(final_lrs, final_wds, c=final_perf, cmap='viridis', s=100)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Weight Decay')
    ax4.set_title('Final Hyperparameters vs Performance')
    plt.colorbar(scatter, ax=ax4, label='Accuracy (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('pbt_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_pbt_experiment()