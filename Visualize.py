import matplotlib.pyplot as plt


def read_results(file_path):
    """Parse log file lines into epoch metrics."""
    parsed = {"epoch": [], "train_acc": [], "test_acc": [], "loss": []}
    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("=") or line.startswith("run"):
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4 or not parts[0].lower().startswith("epoch"):
                continue

            try:
                epoch_token = parts[0].split()[1]
                epoch = int(epoch_token.split("/")[0])
                loss = float(parts[1].split()[1])
                train_acc = float(parts[2].split()[1].rstrip("%"))
                test_acc = float(parts[3].split()[1].rstrip("%"))
            except (IndexError, ValueError):
                # Skip malformed lines instead of crashing.
                continue

            parsed["epoch"].append(epoch)
            parsed["loss"].append(loss)
            parsed["train_acc"].append(train_acc)
            parsed["test_acc"].append(test_acc)

    return parsed


def plot_results(results):
    if not results["epoch"]:
        raise ValueError("No training results found to plot.")

    epochs = results["epoch"]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    train_line = ax1.plot(epochs, results["train_acc"], label="Train Acc", color="tab:blue")
    test_line = ax1.plot(epochs, results["test_acc"], label="Test Acc", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", linewidth=0.5)

    loss_line = []
    if results["loss"]:
        ax2 = ax1.twinx()
        loss_line = ax2.plot(epochs, results["loss"], label="Loss", color="tab:red", linestyle=":")
        ax2.set_ylabel("Loss")

    lines = train_line + test_line + loss_line
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Training Progress")
    plt.tight_layout()
    plt.show()


def main():
    results_file = "./BaseTraining.txt"
    results = read_results(results_file)
    plot_results(results)


if __name__ == "__main__":
    main()
