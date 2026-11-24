"""Small opt-in CUDA validation benchmark for an authorized GPU allocation.

Uses an already-installed PyTorch environment. No downloads, package installs,
server processes, or uploads. Measures matrix multiplication, not LLM serving.
"""
import argparse
import json
import shutil
import subprocess
import sys


def validate(run=False, device_index=0):
    result = {"scope": "single_gpu_matrix_multiplication_validation",
              "nvcc_available": shutil.which("nvcc") is not None,
              "python": sys.version.split()[0], "executed": False}
    if result["nvcc_available"]:
        try:
            output = subprocess.run(["nvcc", "--version"], capture_output=True,
                                    text=True, timeout=10, check=True)
            result["nvcc_version"] = output.stdout.strip().splitlines()[-1]
        except (OSError, subprocess.SubprocessError, IndexError):
            result["nvcc_version"] = "unavailable"
    try:
        import torch
    except (ImportError, OSError):
        result["error"] = "PyTorch is not available in this Python environment."
        return result
    result.update({"torch_version": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
                   "cuda_available": torch.cuda.is_available()})
    if not result["cuda_available"]:
        result["error"] = "PyTorch cannot access CUDA in this environment."
        return result
    result["visible_cuda_devices"] = torch.cuda.device_count()
    if not 0 <= device_index < result["visible_cuda_devices"]:
        result["error"] = "Selected device is not visible to PyTorch."
        return result
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    result.update({"device": properties.name, "compute_capability": [properties.major, properties.minor],
                   "total_memory_mib": round(properties.total_memory / 2**20, 1)})
    if not run:
        return result
    free, _ = torch.cuda.mem_get_info(device)
    if free < 512 * 2**20:
        result["error"] = "Less than 512 MiB free; validation skipped."
        return result
    # Tiny deterministic workload: three 1024-square FP32 matrices are 12 MiB.
    # The CUDA runtime may require additional context memory.
    n = 1024
    repeats = 10
    torch.manual_seed(42)
    torch.set_num_threads(1)
    a_cpu = torch.rand((n, n), dtype=torch.float32)
    b_cpu = torch.rand((n, n), dtype=torch.float32)
    expected = a_cpu @ b_cpu
    with torch.inference_mode(), torch.cuda.device(device):
        a, b = a_cpu.to(device), b_cpu.to(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        actual = a @ b
        torch.cuda.synchronize(device)
        correct = torch.allclose(actual.cpu(), expected, rtol=1e-4, atol=1e-3)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            actual = a @ b
            stop.record()
            stop.synchronize()
            timings.append(start.elapsed_time(stop))
        result.update({
            "executed": True, "correct": bool(correct), "matrix_size": n,
            "dtype": "float32", "repeats": repeats, "gpu_compute_ms": timings,
            "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
            "note": "CUDA event timings exclude data transfer and do not measure LLM or end-to-end service performance.",
        })
    if not correct:
        result["error"] = "CPU/GPU numerical consistency check failed."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run a small computation on an authorized GPU")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    report = validate(args.run, args.device)
    print(json.dumps(report, indent=2))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
