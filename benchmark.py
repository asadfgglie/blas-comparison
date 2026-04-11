import subprocess
import re
import matplotlib.pyplot as plt

EXECUTABLES = [
    "main_cu",
    "main_netlib",
    "main_oneMKL",
    "main_oneMKL_one_thread_process",
    "main_openblas",
    "main_openblas_one_thread_process",
    "main_optimize"
]

SIZES = [100, 500, 1000, 4000]
LOG_FILE = "benchmark.log"
PLOT_FILE = "benchmark_plot.png"
BUILD_DIR = "cmake-build-release"

def run_benchmarks():
    print(f"Starting benchmarks. Logs will be saved to {LOG_FILE}.")
    with open(LOG_FILE, "w") as log:
        for size in SIZES:
            for exe in EXECUTABLES:
                cmd = [f"./{BUILD_DIR}/{exe}", str(size)]
                print(f"Running {' '.join(cmd)}...")
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    # Some output might be in stderr or stdout
                    output = result.stdout + "\n" + result.stderr
                    log.write(f"=== {exe} {size} ===\n")
                    log.write(output)
                    log.write("\n")
                except FileNotFoundError:
                    print(f"Executable {exe} not found in {BUILD_DIR}!")

def parse_logs():
    print(f"Parsing logs from {LOG_FILE}...")
    data = {size: {exe: {} for exe in EXECUTABLES} for size in SIZES}

    current_exe = None
    current_size = None

    with open(LOG_FILE, "r") as log:
        for line in log:
            header_match = re.match(r"=== (\S+) (\d+) ===", line)
            if header_match:
                current_exe = header_match.group(1)
                current_size = int(header_match.group(2))
                continue

            if current_exe and current_size:
                # Regex patterns that handle both CPU and GPU string formats
                m1 = re.search(r"gen SPD timer(?: \(GPU\))?:\s*([\d.]+)s", line)
                if m1:
                    data[current_size][current_exe]['gen_spd'] = float(m1.group(1))

                m2 = re.search(r"gen b vector timer(?: \(GPU\))?:\s*([\d.]+)s", line)
                if m2:
                    data[current_size][current_exe]['gen_vec'] = float(m2.group(1))

                m3 = re.search(r"steepest_descent timer(?: \(GPU\))?:\s*([\d.]+)s,\s*iter:\s*(\d+)", line)
                if m3:
                    t = float(m3.group(1))
                    it = int(m3.group(2))
                    data[current_size][current_exe]['sd_time'] = t
                    data[current_size][current_exe]['sd_iter'] = it
                    if it > 0:
                        data[current_size][current_exe]['sd_norm'] = t / it
                    else:
                        data[current_size][current_exe]['sd_norm'] = 0.0

    return data

def plot_results(data):
    print("Generating plots...")
    # 4 rows (metrics) x 4 columns (sizes)
    fig, axes = plt.subplots(4, 4, figsize=(24, 18))
    metrics = [
        ('gen_spd', 'Gen SPD Timer (s)'),
        ('gen_vec', 'Gen b Vector Timer (s)'),
        ('sd_time', 'Steepest Descent Timer (s)'),
        ('sd_norm', 'Steepest Descent Normalized (s/iter)')
    ]

    for row_idx, (metric_key, metric_title) in enumerate(metrics):
        for col_idx, size in enumerate(SIZES):
            ax = axes[row_idx, col_idx]

            x_labels = []
            y_values = []

            for exe in EXECUTABLES:
                val = data[size][exe].get(metric_key, 0)
                x_labels.append(exe)
                y_values.append(val)

            bars = ax.bar(x_labels, y_values, color='#4C72B0')
            ax.set_title(f"n={size}")
            if col_idx == 0:
                ax.set_ylabel(metric_title, fontsize=12)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1e}',
                        ha='center', va='bottom', fontsize=8)

            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Plot completely generated and saved to {PLOT_FILE}")

if __name__ == "__main__":
    run_benchmarks()
    data = parse_logs()
    plot_results(data)

