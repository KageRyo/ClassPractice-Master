import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY_MIN_NS = 1.0
CPU_INFO = "Intel Core i5-14500"
MEM_INFO = "L1=32~48KB, L2=11.5MB, L3=24MB, RAM=8GB"


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


def read_tlb_assoc_csv(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("TLB associativity CSV is empty")

    header = [cell.strip() for cell in rows[0] if cell.strip()]
    if len(header) < 2 or header[0].lower() != "ways":
        raise ValueError("Invalid TLB associativity CSV header")

    stride_labels = header[1:]
    ways = []
    series = {label: [] for label in stride_labels}

    for row in rows[1:]:
        cleaned = [cell.strip() for cell in row if cell.strip()]
        if len(cleaned) < 2:
            continue

        try:
            w = int(cleaned[0])
        except ValueError:
            continue

        values = []
        ok = True
        for token in cleaned[1:1 + len(stride_labels)]:
            try:
                values.append(float(token))
            except ValueError:
                ok = False
                break
        if not ok or not values:
            continue

        # Pad if row has fewer columns than header.
        while len(values) < len(stride_labels):
            values.append(float("nan"))

        ways.append(w)
        for idx, label in enumerate(stride_labels):
            series[label].append(values[idx])

    if not ways:
        raise ValueError("No valid rows in TLB associativity CSV")

    return stride_labels, ways, series


def is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def detect_page_stride(stride_bytes, array_bytes, matrix) -> int:
    """Heuristically find likely page-size stride from latency jumps on large arrays."""
    if not stride_bytes or not matrix:
        return 4096

    # Prefer large working sets where TLB/cache transitions are clearer.
    large_indices = [i for i, b in enumerate(array_bytes) if b >= 8 * 1024 * 1024]
    if not large_indices:
        large_indices = list(range(max(0, len(array_bytes) // 2), len(array_bytes)))

    avg_by_stride = []
    for col in range(len(stride_bytes)):
        vals = [matrix[row][col] for row in large_indices if col < len(matrix[row]) and is_finite_number(matrix[row][col])]
        avg_by_stride.append(sum(vals) / len(vals) if vals else float("nan"))

    best_idx = None
    best_ratio = 1.0
    for i in range(1, len(stride_bytes)):
        s = stride_bytes[i]
        if s < 512 or s > 65536:
            continue
        prev_v = avg_by_stride[i - 1]
        cur_v = avg_by_stride[i]
        if not (is_finite_number(prev_v) and is_finite_number(cur_v)):
            continue
        if prev_v <= 0:
            continue
        ratio = cur_v / prev_v
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_idx is None:
        return 4096
    return stride_bytes[best_idx]


def estimate_tlb_entries_and_penalty(array_bytes, stride_bytes, matrix, page_stride_bytes: int):
    if not array_bytes or not stride_bytes or not matrix:
        return None, None, None

    # Use the column nearest to page-sized stride.
    col_idx = min(range(len(stride_bytes)), key=lambda i: abs(stride_bytes[i] - page_stride_bytes))

    pages = []
    latencies = []
    for row_idx, arr_b in enumerate(array_bytes):
        if col_idx >= len(matrix[row_idx]):
            continue
        v = matrix[row_idx][col_idx]
        if not is_finite_number(v):
            continue
        pages.append(max(arr_b / float(page_stride_bytes), 1.0))
        latencies.append(v)

    if len(pages) < 4:
        return col_idx, None, None

    # Knee: strongest multiplicative rise between neighboring points.
    first_significant_knee = None
    best_knee_i = None
    best_score = 0.0
    for i in range(1, len(latencies)):
        # Ignore very small page working sets that are dominated by cache effects.
        if pages[i] < 64:
            continue
        prev_v = latencies[i - 1]
        cur_v = latencies[i]
        if prev_v <= 0:
            continue
        ratio = cur_v / prev_v
        delta = cur_v - prev_v
        if first_significant_knee is None and ratio >= 1.5 and delta >= 1.0:
            first_significant_knee = i
        score = (ratio - 1.0) * cur_v
        if score > best_score:
            best_score = score
            best_knee_i = i

    knee_i = first_significant_knee if first_significant_knee is not None else best_knee_i
    if knee_i is None:
        return col_idx, None, None

    tlb_entries_est = int(round(pages[knee_i]))

    low_plateau = [latencies[i] for i in range(len(latencies)) if pages[i] <= tlb_entries_est]
    high_plateau = [latencies[i] for i in range(len(latencies)) if pages[i] >= max(tlb_entries_est * 4, 256)]

    if not low_plateau:
        low_plateau = latencies[: max(1, knee_i)]
    if not high_plateau:
        high_plateau = latencies[knee_i:]

    low_mean = sum(low_plateau) / len(low_plateau)
    high_mean = sum(high_plateau) / len(high_plateau)
    penalty = max(high_mean - low_mean, 0.0)

    return col_idx, tlb_entries_est, penalty


def estimate_tlb_associativity(ways, series) -> int | None:
    if not ways or not series:
        return None

    avg_latency = []
    for i in range(len(ways)):
        vals = [series[label][i] for label in series if i < len(series[label]) and is_finite_number(series[label][i])]
        if not vals:
            avg_latency.append(float("nan"))
        else:
            avg_latency.append(sum(vals) / len(vals))

    finite_pairs = [(w, v) for w, v in zip(ways, avg_latency) if is_finite_number(v)]
    if len(finite_pairs) < 4:
        return None

    # Ignore the very first points, which can be noisy due to tiny loop bodies.
    candidate_pairs = [pair for pair in finite_pairs if pair[0] >= 6]
    if len(candidate_pairs) < 3:
        candidate_pairs = finite_pairs

    # Detect first stable jump. If transition happens at way N,
    # associativity is approximately (N - 1)-way.
    jump_way = None
    for i in range(1, len(candidate_pairs) - 1):
        w, cur_v = candidate_pairs[i]
        _, prev_v = candidate_pairs[i - 1]
        _, next_v = candidate_pairs[i + 1]
        delta = cur_v - prev_v
        if delta >= 1.0 and cur_v >= prev_v * 1.25 and next_v >= prev_v * 1.20:
            jump_way = w
            break

    if jump_way is not None and jump_way >= 2:
        return jump_way - 1

    return None


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
        # Clamp display floor to 1ns so the log-scale plot matches textbook-style readability.
        ys = [max(vy, DISPLAY_MIN_NS) if math.isfinite(vy) else float("nan") for vy in y]
        ax.plot(x, ys, marker="o", linewidth=1.1, markersize=3, label=row_label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Stride Bytes (log2 scale)")
    ax.set_ylabel("Latency (ns/load, log scale)")
    ax.set_title(f"Elapsed Time vs Stride (Log-Log)\n{CPU_INFO} | {MEM_INFO}")
    ax.set_ylim(bottom=DISPLAY_MIN_NS)

    # Keep readable tick labels on x-axis while using log2 spacing.
    ax.set_xticks(stride_bytes)
    ax.set_xticklabels([bytes_to_label(b) for b in stride_bytes], rotation=45, ha="right")

    ax.legend(
        title="Array Size",
        fontsize=7,
        title_fontsize=9,
        loc="upper right",
        ncol=2,
        frameon=True,
    )

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_textbook_style(stride_labels, array_labels, matrix, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Textbook-like view: x as array-size categories, y as linear latency.
    x = list(range(len(array_labels)))
    target_strides = ["16B", "64B", "256B", "1K"]

    for stride_label in target_strides:
        if stride_label not in stride_labels:
            continue
        col_idx = stride_labels.index(stride_label)
        y = [matrix[row_idx][col_idx] for row_idx in range(len(array_labels))]
        ax.plot(x, y, marker="o", linewidth=1.4, markersize=4, label=f"Stride {stride_label}")

    ax.set_xticks(x)
    ax.set_xticklabels(array_labels, rotation=45, ha="right")
    ax.set_xlabel("Array Size")
    ax.set_ylabel("Latency (ns/load)")
    ax.set_title(f"Memory Access Latency (Textbook Style)\n{CPU_INFO} | {MEM_INFO}")
    ax.legend(title="Stride Size", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_page_size_probe(stride_bytes, array_labels, array_bytes, matrix, page_stride_bytes: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Focus on larger arrays where the page-size transition is easier to observe.
    large_rows = [i for i, b in enumerate(array_bytes) if b >= 4 * 1024 * 1024]
    if len(large_rows) > 5:
        # Keep plot readable by taking a spread of large rows.
        step = max(1, len(large_rows) // 5)
        large_rows = large_rows[::step]

    for row_idx in large_rows:
        y = [max(v, DISPLAY_MIN_NS) if is_finite_number(v) else float("nan") for v in matrix[row_idx]]
        ax.plot(stride_bytes, y, marker="o", linewidth=1.2, markersize=3.5, label=array_labels[row_idx])

    ax.axvline(page_stride_bytes, color="red", linestyle="--", linewidth=1.2, label=f"Estimated page size: {bytes_to_label(page_stride_bytes)}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Stride Bytes (log2 scale)")
    ax.set_ylabel("Latency (ns/load, log scale)")
    ax.set_title("Page-Size Probe from Memory Mountain Data")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Array Size", fontsize=8, title_fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tlb_entries_probe(array_bytes, stride_bytes, matrix, page_stride_bytes: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))

    col_idx = min(range(len(stride_bytes)), key=lambda i: abs(stride_bytes[i] - page_stride_bytes))
    pages = []
    latencies = []
    for row_idx, arr_b in enumerate(array_bytes):
        if col_idx >= len(matrix[row_idx]):
            continue
        v = matrix[row_idx][col_idx]
        if not is_finite_number(v):
            continue
        pages.append(max(arr_b / float(page_stride_bytes), 1.0))
        latencies.append(max(v, DISPLAY_MIN_NS))

    ax.plot(pages, latencies, marker="o", linewidth=1.4, markersize=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"Pages Touched (ArrayBytes / {bytes_to_label(page_stride_bytes)})")
    ax.set_ylabel("Latency (ns/load, log scale)")
    ax.set_title("TLB Entries / Miss Penalty Probe")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tlb_associativity_probe(stride_bytes, array_labels, array_bytes, matrix, page_stride_bytes: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Explore only strides that are integer multiples of the estimated page size.
    page_multiple_indices = [
        i for i, s in enumerate(stride_bytes)
        if s >= page_stride_bytes and (s % page_stride_bytes == 0)
    ]

    if not page_multiple_indices:
        page_multiple_indices = [i for i, s in enumerate(stride_bytes) if s >= page_stride_bytes]

    # Use larger array sizes to increase chance of visible TLB conflicts.
    target_rows = [i for i, b in enumerate(array_bytes) if b >= 8 * 1024 * 1024]
    if len(target_rows) > 4:
        step = max(1, len(target_rows) // 4)
        target_rows = target_rows[::step]

    x = [stride_bytes[i] / float(page_stride_bytes) for i in page_multiple_indices]
    for row_idx in target_rows:
        y = [
            max(matrix[row_idx][i], DISPLAY_MIN_NS)
            if i < len(matrix[row_idx]) and is_finite_number(matrix[row_idx][i])
            else float("nan")
            for i in page_multiple_indices
        ]
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=3.5, label=array_labels[row_idx])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Stride / PageSize (multiples, log2 scale)")
    ax.set_ylabel("Latency (ns/load, log scale)")
    ax.set_title("TLB Associativity Probe (Exploratory)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Array Size", fontsize=8, title_fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tlb_associativity_conflict(stride_labels, ways, series, assoc_est, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))

    for label in stride_labels:
        y = [max(v, DISPLAY_MIN_NS) if is_finite_number(v) else float("nan") for v in series[label]]
        ax.plot(ways, y, marker="o", linewidth=1.4, markersize=4, label=f"Stride {label}")

    if assoc_est is not None:
        ax.axvline(assoc_est, color="green", linestyle="--", linewidth=1.3, label=f"Estimated associativity: {assoc_est}-way")

    ax.set_xlabel("Ways Needed (pages mapped to same TLB set)")
    ax.set_ylabel("Latency (ns/access)")
    ax.set_title("TLB Associativity Measurement (Dedicated Conflict Benchmark)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_tlb_summary(out_path: Path, page_stride_bytes: int, tlb_entries_est, tlb_miss_penalty_est, tlb_assoc_est):
    lines = [
        "TLB Analysis Summary (heuristic)",
        "================================",
        f"Estimated page size: {bytes_to_label(page_stride_bytes)} ({page_stride_bytes} bytes)",
    ]

    if tlb_entries_est is None:
        lines.append("Estimated TLB entries: inconclusive from current dataset")
    else:
        lines.append(f"Estimated TLB entries: ~{tlb_entries_est}")

    if tlb_miss_penalty_est is None:
        lines.append("Estimated TLB miss penalty: inconclusive from current dataset")
    else:
        lines.append(f"Estimated TLB miss penalty: ~{tlb_miss_penalty_est:.2f} ns/load")

    if tlb_assoc_est is None:
        lines.append("Estimated TLB associativity: inconclusive (please run dedicated conflict benchmark)")
    else:
        lines.append(f"Estimated TLB associativity: ~{tlb_assoc_est}-way")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_memory_benchmark.py <memory_benchmark.csv>")
        raise SystemExit(1)

    csv_path = Path(sys.argv[1]).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = csv_path.parent

    stride_labels, stride_bytes, array_labels, array_bytes, matrix = read_benchmark_csv(csv_path)

    heatmap_path = outdir / "memory_heatmap.png"
    lines_path = outdir / "memory_lines_by_array_size.png"
    textbook_path = outdir / "memory_textbook_style.png"
    page_probe_path = outdir / "tlb_page_size_probe.png"
    entries_probe_path = outdir / "tlb_entries_probe.png"
    assoc_probe_path = outdir / "tlb_associativity_probe.png"
    assoc_csv_path = outdir / "tlb_assoc_benchmark.csv"
    assoc_conflict_path = outdir / "tlb_associativity_conflict.png"
    tlb_summary_path = outdir / "tlb_analysis_summary.txt"

    plot_heatmap(stride_bytes, array_bytes, matrix, heatmap_path)
    plot_lines_by_array_size(stride_bytes, array_labels, matrix, lines_path)
    plot_textbook_style(stride_labels, array_labels, matrix, textbook_path)

    page_stride_bytes = detect_page_stride(stride_bytes, array_bytes, matrix)
    tlb_col_idx, tlb_entries_est, tlb_miss_penalty_est = estimate_tlb_entries_and_penalty(
        array_bytes,
        stride_bytes,
        matrix,
        page_stride_bytes,
    )
    _ = tlb_col_idx

    plot_page_size_probe(stride_bytes, array_labels, array_bytes, matrix, page_stride_bytes, page_probe_path)
    plot_tlb_entries_probe(array_bytes, stride_bytes, matrix, page_stride_bytes, entries_probe_path)
    plot_tlb_associativity_probe(stride_bytes, array_labels, array_bytes, matrix, page_stride_bytes, assoc_probe_path)

    tlb_assoc_est = None
    if assoc_csv_path.exists():
        stride_labels_assoc, ways_assoc, series_assoc = read_tlb_assoc_csv(assoc_csv_path)
        tlb_assoc_est = estimate_tlb_associativity(ways_assoc, series_assoc)
        plot_tlb_associativity_conflict(
            stride_labels_assoc,
            ways_assoc,
            series_assoc,
            tlb_assoc_est,
            assoc_conflict_path,
        )

    write_tlb_summary(
        tlb_summary_path,
        page_stride_bytes,
        tlb_entries_est,
        tlb_miss_penalty_est,
        tlb_assoc_est,
    )

    print(f"Saved: {heatmap_path}")
    print(f"Saved: {lines_path}")
    print(f"Saved: {textbook_path}")
    print(f"Saved: {page_probe_path}")
    print(f"Saved: {entries_probe_path}")
    print(f"Saved: {assoc_probe_path}")
    if assoc_csv_path.exists():
        print(f"Saved: {assoc_conflict_path}")
    print(f"Saved: {tlb_summary_path}")


if __name__ == "__main__":
    main()
