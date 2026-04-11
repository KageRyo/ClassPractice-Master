import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_size_token(token: str) -> int:
    """Convert labels like 512B, 16K, 24M, 1G to bytes."""
    token = token.strip().upper()
    if not token:
        raise ValueError("Empty size token")

    unit = token[-1]
    number_part = token[:-1]

    if unit not in {"B", "K", "M", "G"}:
        raise ValueError(f"Unsupported size unit in token: {token}")

    value = float(number_part)
    scale = {
        "B": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }[unit]
    return int(value * scale)


def bytes_to_label(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024**2:
        return f"{num_bytes // 1024}K"
    if num_bytes < 1024**3:
        return f"{num_bytes // (1024**2)}M"
    return f"{num_bytes // (1024**3)}G"


def safe_log2(v: int) -> float:
    return math.log2(max(v, 1))


def read_benchmark_csv(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("CSV is empty")

    # Header format from the C benchmark:
    #  ,4B,8B,16B,...
    header = [cell.strip() for cell in rows[0] if cell.strip()]
    stride_labels = header
    stride_bytes = [parse_size_token(x) for x in stride_labels]

    array_labels = []
    array_bytes = []
    matrix = []

    for row in rows[1:]:
        cleaned = [cell.strip() for cell in row if cell.strip()]
        if not cleaned:
            continue

        # Skip non-data lines if user accidentally merged stderr into CSV.
        try:
            row_array_label = cleaned[0]
            row_array_bytes = parse_size_token(row_array_label)
        except Exception:
            continue

        values_raw = cleaned[1:]
        try:
            values = [float(v) for v in values_raw]
        except ValueError:
            continue

        if not values:
            continue

        # Keep width aligned with header length.
        values = values[: len(stride_bytes)]
        if len(values) < len(stride_bytes):
            values.extend([float("nan")] * (len(stride_bytes) - len(values)))

        array_labels.append(row_array_label)
        array_bytes.append(row_array_bytes)
        matrix.append(values)

    if not matrix:
        raise ValueError("No valid benchmark rows found in CSV")

    return stride_labels, stride_bytes, array_labels, array_bytes, matrix


def plot_heatmap(stride_bytes, array_bytes, matrix, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")

    x_ticks = list(range(len(stride_bytes)))
    y_ticks = list(range(len(array_bytes)))

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([bytes_to_label(b) for b in stride_bytes], rotation=45, ha="right")
    ax.set_yticklabels([bytes_to_label(b) for b in array_bytes])

    ax.set_xlabel("Stride Size")
    ax.set_ylabel("Array Size")
    ax.set_title("Memory Access Latency Heatmap (ns/load)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Latency (ns/load)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lines_by_array_size(stride_bytes, array_labels, matrix, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))

    x = stride_bytes

    for row_label, y in zip(array_labels, matrix):
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=3, label=row_label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Stride Bytes (log2 scale)")
    ax.set_ylabel("Latency (ns/load, log scale)")
    ax.set_title("Latency vs Stride (Log-Log, Grouped by Array Size)")

    # Keep readable tick labels on x-axis while using log2 spacing.
    ax.set_xticks(stride_bytes)
    ax.set_xticklabels([bytes_to_label(b) for b in stride_bytes], rotation=45, ha="right")

    # Avoid unreadable legends with too many lines.
    if len(array_labels) <= 12:
        ax.legend(title="Array Size", fontsize=8)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_memory_benchmark.py <memory_benchmark.csv>")
        raise SystemExit(1)

    csv_path = Path(sys.argv[1]).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = csv_path.parent

    _, stride_bytes, array_labels, array_bytes, matrix = read_benchmark_csv(csv_path)

    heatmap_path = outdir / "memory_heatmap.png"
    lines_path = outdir / "memory_lines_by_array_size.png"

    plot_heatmap(stride_bytes, array_bytes, matrix, heatmap_path)
    plot_lines_by_array_size(stride_bytes, array_labels, matrix, lines_path)

    print(f"Saved: {heatmap_path}")
    print(f"Saved: {lines_path}")


if __name__ == "__main__":
    main()
