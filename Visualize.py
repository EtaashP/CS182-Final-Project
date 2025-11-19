import os
import re
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


def empty_results():
    return {"epoch": [], "train_acc": [], "test_acc": [], "loss": []}


RUN_HEADER_RE = re.compile(
    r"^(Phase\s+\d+\s+\([^)]+\))\s+run\s+(\d+)/(\d+)\s*\|\s*(.+)$", re.IGNORECASE
)


def parse_phase_runs(file_path):
    """Return all runs grouped by phase name."""
    phases = {}
    current_run = None
    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            run_match = RUN_HEADER_RE.match(line)
            if run_match:
                phase_label = run_match.group(1)
                run_index = int(run_match.group(2))
                total_runs = int(run_match.group(3))
                hyperparams = run_match.group(4).strip()
                current_run = {
                    "phase": phase_label,
                    "run_index": run_index,
                    "total_runs": total_runs,
                    "hyperparams": hyperparams,
                    "epochs": [],
                }
                phases.setdefault(phase_label, []).append(current_run)
                continue

            if not line.lower().startswith("epoch") or current_run is None:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4:
                continue

            try:
                epoch_token = parts[0].split()[1]
                epoch = int(epoch_token.split("/")[0])
                loss = float(parts[1].split()[1])
                train_acc = float(parts[2].split()[1].rstrip("%"))
                test_acc = float(parts[3].split()[1].rstrip("%"))
            except (IndexError, ValueError):
                continue

            current_run["epochs"].append(
                {
                    "epoch_in_run": epoch,
                    "loss": loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
            )

    return phases


def select_best_run(runs):
    """Pick the run that reached the highest test accuracy."""
    best_run = None
    best_score = float("-inf")
    for run in runs:
        if not run["epochs"]:
            continue
        run_best = max(epoch["test_acc"] for epoch in run["epochs"])
        if run_best > best_score:
            best_score = run_best
            best_run = run
    return best_run


def select_global_best_run(phases):
    """Return the single best run across all phases."""
    best_run = None
    best_score = float("-inf")
    for runs in phases.values():
        for run in runs:
            if not run["epochs"]:
                continue
            run_best = max(epoch["test_acc"] for epoch in run["epochs"])
            if run_best > best_score:
                best_score = run_best
                best_run = run
    return best_run


def _phase_number(phase_label):
    match = re.search(r"Phase\s+(\d+)", phase_label)
    return int(match.group(1)) if match else 0


def compute_phase_offsets(ordered_phases, best_runs_by_phase):
    """Determine where each phase should start on a global epoch axis."""
    offsets = {}
    offset = 0
    for phase_label, _ in ordered_phases:
        offsets[phase_label] = offset
        best_run = best_runs_by_phase.get(phase_label)
        if best_run:
            offset += len(best_run["epochs"])
    return offsets


def build_continuous_results(run_sequence):
    """Combine consecutive runs so epochs keep increasing."""
    combined = {"epoch": [], "train_acc": [], "test_acc": [], "loss": []}
    segments = []
    epoch_offset = 0

    for phase_label, run in run_sequence:
        start_epoch = epoch_offset + 1
        for entry in run["epochs"]:
            epoch_offset += 1
            combined["epoch"].append(epoch_offset)
            combined["loss"].append(entry["loss"])
            combined["train_acc"].append(entry["train_acc"])
            combined["test_acc"].append(entry["test_acc"])
        segments.append(
            {
                "label": phase_label,
                "start": start_epoch,
                "end": epoch_offset,
                "meta": run["hyperparams"],
            }
        )

    return combined, segments


def plot_results(results, title, segments=None, ax=None):
    if not results["epoch"]:
        raise ValueError("No training results found to plot.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.figure

    epochs = results["epoch"]

    train_line = ax.plot(epochs, results["train_acc"], label="Train Acc", color="tab:blue")
    test_line = ax.plot(epochs, results["test_acc"], label="Test Acc", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.5)

    loss_line = []
    if results["loss"]:
        ax2 = ax.twinx()
        loss_line = ax2.plot(epochs, results["loss"], label="Loss", color="tab:red", linestyle=":")
        ax2.set_ylabel("Loss")

    if segments:
        for segment in segments[1:]:
            ax.axvline(segment["start"] - 1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        for segment in segments:
            midpoint = (segment["start"] + segment["end"]) / 2
            ax.text(
                midpoint,
                3,
                segment["label"],
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.7,
            )

    lines = train_line + test_line + loss_line
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="best")
    ax.set_title(title)

    if created_fig:
        fig.tight_layout()
        plt.show()

    return ax


def normalize_epoch_list(epoch_list):
    if not epoch_list:
        return []
    start = epoch_list[0]
    return [epoch - start for epoch in epoch_list]


def plot_all_runs(ordered_phases, best_runs_by_phase, segments, ax=None):
    if not ordered_phases:
        return None

    offsets = compute_phase_offsets(ordered_phases, best_runs_by_phase)
    global_best = select_global_best_run(dict(ordered_phases))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.figure
    other_label_added = False

    for phase_label, runs in ordered_phases:
        offset = offsets.get(phase_label, 0)
        for run in runs:
            if not run["epochs"]:
                continue
            epochs = [offset + idx for idx in range(1, len(run["epochs"]) + 1)]
            test_acc = [entry["test_acc"] for entry in run["epochs"]]

            color = "lightgray"
            linewidth = 0.8
            alpha = 0.35
            label = None

            if run is global_best:
                color = "tab:red"
                linewidth = 2.5
                alpha = 0.95
                label = f"Best overall ({run['phase']} | {run['hyperparams']})"
            elif not other_label_added:
                label = "Other runs (test acc)"
                other_label_added = True

            ax.plot(epochs, test_acc, color=color, linewidth=linewidth, alpha=alpha, label=label)

    if segments:
        for segment in segments[1:]:
            ax.axvline(segment["start"] - 1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        for segment in segments:
            midpoint = (segment["start"] + segment["end"]) / 2
            ax.text(
                midpoint,
                95,
                segment["label"],
                ha="center",
                va="top",
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right")
    ax.set_title("All Phase Runs (best highlighted)")

    if created_fig:
        fig.tight_layout()
        plt.show()

    return ax


def prepare_phase_data(file_path):
    if not os.path.exists(file_path):
        return None

    phases = parse_phase_runs(file_path)
    if not phases:
        return None

    ordered_phases = sorted(phases.items(), key=lambda item: _phase_number(item[0]))
    best_sequence = []
    best_runs_by_phase = {}
    for phase_label, runs in ordered_phases:
        best_run = select_best_run(runs)
        if best_run:
            best_sequence.append((phase_label, best_run))
            best_runs_by_phase[phase_label] = best_run

    if not best_sequence:
        return None

    combined, segments = build_continuous_results(best_sequence)
    return {
        "ordered_phases": ordered_phases,
        "best_runs_by_phase": best_runs_by_phase,
        "combined": combined,
        "segments": segments,
    }


def _rebase_segments(segments, shift):
    if not segments:
        return []
    rebased = []
    for segment in segments:
        rebased.append(
            {
                "label": segment["label"],
                "start": segment["start"] - shift,
                "end": segment["end"] - shift,
                "meta": segment["meta"],
            }
        )
    return rebased


def plot_base_and_phase(ax, base_results, phase_data):
    base_epochs = normalize_epoch_list(base_results["epoch"])
    phase_epochs = []
    rebased_segments = []
    if phase_data and phase_data["combined"]["epoch"]:
        phase_epochs = normalize_epoch_list(phase_data["combined"]["epoch"])
        rebased_segments = _rebase_segments(phase_data["segments"], phase_data["combined"]["epoch"][0])

    loss_ax = None
    if base_results["loss"] or (phase_data and phase_data["combined"]["loss"]):
        loss_ax = ax.twinx()
        loss_ax.set_ylabel("Loss")

    lines = []
    labels = []

    if base_epochs:
        lines += ax.plot(base_epochs, base_results["train_acc"], label="Base Train", color="tab:blue")
        lines += ax.plot(base_epochs, base_results["test_acc"], label="Base Test", color="tab:cyan")
        labels.extend(["Base Train", "Base Test"])
        if loss_ax:
            loss_lines = loss_ax.plot(
                base_epochs,
                base_results["loss"],
                label="Base Loss",
                color="tab:blue",
                linestyle="--",
            )
            lines += loss_lines
            labels.append("Base Loss")

    if phase_epochs:
        lines += ax.plot(phase_epochs, phase_data["combined"]["train_acc"], label="Phase Train", color="tab:orange")
        lines += ax.plot(phase_epochs, phase_data["combined"]["test_acc"], label="Phase Test", color="tab:red")
        labels.extend(["Phase Train", "Phase Test"])
        if loss_ax:
            loss_lines = loss_ax.plot(
                phase_epochs,
                phase_data["combined"]["loss"],
                label="Phase Loss",
                color="tab:orange",
                linestyle=":",
            )
            lines += loss_lines
            labels.append("Phase Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.5)

    if rebased_segments:
        for segment in rebased_segments[1:]:
            ax.axvline(segment["start"], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        for segment in rebased_segments:
            midpoint = (segment["start"] + segment["end"]) / 2
            ax.text(
                midpoint,
                3,
                segment["label"],
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.7,
            )

    if lines:
        handles = lines
        labels = [line.get_label() for line in handles]
        ax.legend(handles, labels, loc="best")
    ax.set_title("Base vs Phase Progress (epochs start at 0)")


def plot_combined_figure(base_results, phase_data):
    has_phase = phase_data is not None and phase_data["combined"]["epoch"]
    total_rows = 2 if has_phase else 1
    fig, axes = plt.subplots(total_rows, 1, figsize=(12, 10 if total_rows == 2 else 5))
    if total_rows == 1:
        axes = [axes]

    plot_base_and_phase(axes[0], base_results, phase_data if has_phase else None)

    if has_phase:
        plot_all_runs(
            phase_data["ordered_phases"],
            phase_data["best_runs_by_phase"],
            phase_data["segments"],
            ax=axes[1],
        )

    fig.tight_layout()
    plt.show()


def main():
    base_file = "./BaseTraining.txt"
    base_results = empty_results()
    if os.path.exists(base_file):
        base_results = read_results(base_file)

    phase_data = prepare_phase_data("./phase_results.txt")

    if not base_results["epoch"] and not phase_data:
        raise SystemExit("No training logs available to visualize.")

    plot_combined_figure(base_results, phase_data)


if __name__ == "__main__":
    main()
