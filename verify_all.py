import os
import subprocess
import sys

EXECUTABLES = [
    "main_cu",
    "main_netlib",
    "main_oneMKL",
    "main_oneMKL_one_thread_process",
    "main_openblas",
    "main_openblas_one_thread_process",
    "main_optimize"
]

BUILD_DIR = "cmake-build-release"
TEST_SIZE = "100"  # A moderate size is enough to test correctness
VERBOSE = "0"

def verify_implementations():
    print(f"Starting correctness verification with matrix size {TEST_SIZE}x{TEST_SIZE}...\n")

    all_passed = True
    for exe in EXECUTABLES:
        exe_path = f"./{BUILD_DIR}/{exe}"
        result_file = f"test_result_{exe}.txt"

        if not os.path.exists(exe_path):
            print(f"{exe}: Executable not found at {exe_path}. Did you build the project?")
            all_passed = False
            continue

        print(f"Testing {exe}...")

        run_cmd = [exe_path, TEST_SIZE, result_file, VERBOSE]
        try:
            subprocess.run(run_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"{exe}: Execution failed.\n{e.stderr}")
            all_passed = False
            continue

        check_cmd = ["uv", "run", "python", "scipy_solver.py", "-f", result_file]
        try:
            check_res = subprocess.run(check_cmd, check=True, capture_output=True, text=True)
            output = check_res.stdout

            if "False" in output:
                print(f"{exe}: Verification FAILED! Error tolerances exceeded.")
                print("--- scipy_solver details ---")
                print(output.strip())
                print("----------------------------\n")
                all_passed = False
            else:
                print(f"{exe}: Passed verification!")

        except subprocess.CalledProcessError as e:
            print(f"{exe}: Failed to run checking script.\n{e.stderr}")
            all_passed = False

        if os.path.exists(result_file):
            os.remove(result_file)

    if all_passed:
        print("\nAll implementations passed the correctness check!")
    else:
        print("\nSome implementations failed the correctness check.")
        sys.exit(1)

if __name__ == "__main__":
    verify_implementations()

