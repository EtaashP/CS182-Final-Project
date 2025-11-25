import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))

# Simple MLP
model = nn.Sequential(
    nn.Linear(384, 1024),
    nn.GELU(),
    nn.Linear(1024, 10)
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for step in range(10):
    x = torch.randn(64, 384, device=device)
    y = torch.randint(0, 10, (64,), device=device)

    opt.zero_grad(set_to_none=True)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    opt.step()

    print(f"step {step} loss={loss.item():.4f}")

