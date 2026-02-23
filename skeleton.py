#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def to_labels(y: np.ndarray) -> np.ndarray:
    """Convert labels to a flat {-1, 1} array."""
    y = np.asarray(y).reshape(-1)
    return np.where(y >= 0, 1, -1).astype(int)


def add_bias(x: np.ndarray) -> np.ndarray:
    """Append a constant 1 feature for the bias term."""
    x = np.asarray(x, dtype=float)
    return np.hstack((x, np.ones((x.shape[0], 1), dtype=float)))


def predict(w: np.ndarray, x_with_bias: np.ndarray) -> np.ndarray:
    """Predict class labels {-1, 1}."""
    return np.where(x_with_bias @ w >= 0, 1, -1)


def accuracy(w: np.ndarray, x_with_bias: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_hat = predict(w, x_with_bias)
    return float(np.mean(y_hat == y))


def evaluate_on_split(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Evaluate a weight vector on arbitrary split data."""
    return accuracy(w, add_bias(x), to_labels(y))


def load_plot_tools():
    """Import plotting dependencies only when they are needed."""
    try:
        import matplotlib.pyplot as plt
        from visualise import visualise
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plotting was requested, but matplotlib is not installed. "
            "Install matplotlib or run with --plot-every 0."
        ) from exc
    return plt, visualise


def pla(
    data: dict[str, np.ndarray],
    plot_every: Optional[int] = 1,
    max_iteration: int = 100,
    use_pocket: bool = False,
    random_update: bool = False,
    pause: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list[float]]:
    """
    Train a perceptron (or pocket perceptron if use_pocket=True).

    Parameters
    ----------
    data:
        Dictionary with "X" and "Y".
    plot_every:
        Plot every N iterations. Use 0 or None to disable plotting.
    max_iteration:
        Maximum number of update iterations.
    use_pocket:
        If True, keep and return the best-so-far weight vector.
    random_update:
        If True, pick a random misclassified sample on each update.
    pause:
        Pause duration for nonblocking plotting.
    seed:
        Random seed used when random_update=True.
    """
    x = np.asarray(data["X"], dtype=float)
    y = to_labels(data["Y"])
    x_with_bias = add_bias(x)

    w = np.zeros(x_with_bias.shape[1], dtype=float)
    best_w = w.copy()
    best_acc = accuracy(best_w, x_with_bias, y)

    fig, ax = None, None
    plot_enabled = bool(plot_every) and x.shape[1] == 2
    plt_mod, visualise_fn = (None, None)
    if plot_enabled:
        try:
            plt_mod, visualise_fn = load_plot_tools()
        except RuntimeError as exc:
            print(exc)
            plot_enabled = False
    elif plot_every and x.shape[1] != 2:
        print("Skipping plots: visualise() supports only 2D inputs.")

    history: list[float] = []
    rng = np.random.default_rng(seed)

    for iteration in range(max_iteration):
        y_hat = predict(w, x_with_bias)
        misclassified = np.where(y_hat != y)[0]
        current_acc = float(np.mean(y_hat == y))
        history.append(current_acc)

        w_for_plot = best_w if use_pocket else w
        if plot_enabled and iteration % plot_every == 0:
            if ax is not None:
                ax.clear()
            fig, ax, _ = visualise_fn(
                w_for_plot,
                x,
                y,
                ax=ax,
                title=f"iter={iteration}, train acc={current_acc:.3f}",
            )
            plt_mod.pause(pause)

        if misclassified.size == 0:
            break

        if random_update:
            sample_index = int(rng.choice(misclassified))
        else:
            sample_index = int(misclassified[0])
        w = w + y[sample_index] * x_with_bias[sample_index]

        if use_pocket:
            current_w_acc = accuracy(w, x_with_bias, y)
            if current_w_acc > best_acc:
                best_acc = current_w_acc
                best_w = w.copy()

    final_w = best_w if use_pocket else w
    final_acc = accuracy(final_w, x_with_bias, y)

    if plot_enabled:
        if ax is not None:
            ax.clear()
        fig, ax, _ = visualise_fn(
            final_w,
            x,
            y,
            ax=ax,
            title=f"final, train acc={final_acc:.3f}",
        )
        plt_mod.pause(pause)

    if plot_every:
        return final_w, final_acc, history
    return final_w, final_acc


def load_npz_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    """Load npz into a plain dictionary."""
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def run_dataset(
    dataset_path: Path,
    max_iteration: int,
    plot_every: int,
    pause: float,
    seed: int,
    use_pocket: bool,
    random_update: bool,
) -> tuple[np.ndarray, float, float]:
    data = load_npz_dataset(dataset_path)
    result = pla(
        data,
        plot_every=plot_every,
        max_iteration=max_iteration,
        use_pocket=use_pocket,
        random_update=random_update,
        pause=pause,
        seed=seed,
    )
    w, train_acc = result[:2]
    test_acc = evaluate_on_split(w, data["X_test"], data["Y_test"])
    return w, train_acc, test_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple PLA and pocket PLA runner.")
    parser.add_argument(
        "--pla-path",
        type=Path,
        default=Path("data/pla.npz"),
        help="Path to pla.npz dataset.",
    )
    parser.add_argument(
        "--pocket-path",
        type=Path,
        default=Path("data/pocket.npz"),
        help="Path to pocket.npz dataset.",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=200,
        help="Maximum number of perceptron updates.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="Plot every N iterations. Use 0 to disable plotting.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Pause (seconds) between plotted iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for random updates.",
    )
    args = parser.parse_args()

    print("=== PLA on pla.npz ===")
    if args.pla_path.exists():
        w_pla, train_acc_pla, test_acc_pla = run_dataset(
            dataset_path=args.pla_path,
            max_iteration=args.max_iteration,
            plot_every=args.plot_every,
            pause=args.pause,
            seed=args.seed,
            use_pocket=False,
            random_update=False,
        )
        print(f"weights: {w_pla}")
        print(f"train accuracy: {train_acc_pla:.3f}")
        print(f"test accuracy:  {test_acc_pla:.3f}")
    else:
        print(f"Skipped: {args.pla_path} not found.")

    print("\n=== Pocket PLA on pocket.npz ===")
    if args.pocket_path.exists():
        w_base, train_base, test_base = run_dataset(
            dataset_path=args.pocket_path,
            max_iteration=args.max_iteration,
            plot_every=args.plot_every,
            pause=args.pause,
            seed=args.seed,
            use_pocket=True,
            random_update=False,
        )
        print("Baseline pocket (first misclassified sample):")
        print(f"weights: {w_base}")
        print(f"train accuracy: {train_base:.3f}")
        print(f"test accuracy:  {test_base:.3f}")

        w_mod, train_mod, test_mod = run_dataset(
            dataset_path=args.pocket_path,
            max_iteration=args.max_iteration,
            plot_every=0,
            pause=args.pause,
            seed=args.seed,
            use_pocket=True,
            random_update=True,
        )
        print("\nSmall modification: random misclassified sample update")
        print(f"weights: {w_mod}")
        print(f"train accuracy: {train_mod:.3f}")
        print(f"test accuracy:  {test_mod:.3f}")
        if test_mod >= test_base:
            print("Result: the small modification performed better (or equal) on test.")
        else:
            print("Result: the baseline performed better on this run.")
    else:
        print(f"Skipped: {args.pocket_path} not found.")

    if args.plot_every > 0:
        try:
            plt_mod, _ = load_plot_tools()
            plt_mod.show()
        except RuntimeError as exc:
            print(exc)


if __name__ == "__main__":
    main()
