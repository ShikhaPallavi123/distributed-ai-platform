"""Read-only GPU capability probe. Run on an authorized GPU node.

No packages installed, no workloads launched, no network calls, no data upload.
Does not collect usernames, hostnames, IP addresses or GPU serial numbers.
"""
import csv
import json
import os
import platform
import shutil
import subprocess
import sys


def probe():
    fields = ["index", "name", "memory.total", "memory.free", "driver_version"]
    executable = shutil.which("nvidia-smi")
    result = {
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "gpus": [],
        "cuda_visible_devices_is_set": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "note": "Physical GPU inventory may exceed your assigned allocation; verify scheduler permissions.",
    }
    if not executable:
        result["error"] = "nvidia-smi not found; run on an allocated NVIDIA GPU node."
        return result
    try:
        process = subprocess.run(
            [executable, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        result["error"] = "GPU query failed; check your allocation and NVIDIA driver."
        return result
    for row in csv.reader(process.stdout.splitlines()):
        if len(row) == len(fields):
            result["gpus"].append(dict(zip(fields, (item.strip() for item in row))))
    result["visible_inventory_count"] = len(result["gpus"])
    if not result["gpus"]:
        result["error"] = "No NVIDIA devices reported."
    return result


if __name__ == "__main__":
    report = probe()
    print(json.dumps(report, indent=2))
    sys.exit(int("error" in report))
