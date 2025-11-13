# jordan_experiment.py

import numpy as np
import matplotlib.pyplot as plt

def run_experiment():
    print("Running Jordan's experiment...")
    # Example: generate and plot a sine wave
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_experiment()