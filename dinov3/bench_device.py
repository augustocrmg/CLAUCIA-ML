"""Mede no dispositivo Android (adb) os .tflite de tflite/arch/ com o
benchmark_model oficial do LiteRT.

Para cada modelo x variante x configuração:
  cpu4 : XNNPACK, 4 threads;  cpu1 : XNNPACK, 1 thread;  gpu : delegate GPU (OpenCL/GL)
3 repetições (10 aquecimentos + 50 inferências cada), pausa de COOLDOWN s entre
elas, temperatura da bateria antes/depois. Saída: device_bench/<serial>.csv
Uso: python bench_device.py [--models dinov3_small mobilenet_v2] [--configs cpu4 gpu]
"""
import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADB = os.path.expanduser("~/.local/android/platform-tools/adb")
BENCH_BIN = os.path.expanduser("~/.local/android/bench/benchmark_model")
REMOTE = "/data/local/tmp/bench"
CONFIGS = {
    "cpu4": "--use_xnnpack=true --num_threads=4",
    "cpu1": "--use_xnnpack=true --num_threads=1",
    "gpu": "--use_gpu=true --gpu_precision_loss_allowed=false",
}
REPEATS = int(os.environ.get("BENCH_REPEATS", 3))
COOLDOWN = int(os.environ.get("BENCH_COOLDOWN", 20))


def adb(*args, timeout=900):
    return subprocess.run([ADB, *args], capture_output=True, text=True, timeout=timeout).stdout


def battery_temp():
    m = re.search(r"temperature: (\d+)", adb("shell", "dumpsys", "battery"))
    return int(m.group(1)) / 10 if m else None


def parse(out):
    r = {}
    m = re.search(r"Inference timings in us: Init: ([\d.e+]+), First inference: ([\d.e+]+), Warmup \(avg\): ([\d.e+]+), Inference \(avg\): ([\d.e+]+)", out)
    if m:
        r.update(init_ms=float(m[1]) / 1e3, first_ms=float(m[2]) / 1e3, warmup_avg_ms=float(m[3]) / 1e3, avg_ms=float(m[4]) / 1e3)
    runs = [l for l in out.splitlines() if "count=" in l and "median=" in l]
    if runs:
        s = runs[-1]
        for k in ["std", "median", "p95", "min", "max"]:
            mm = re.search(rf"{k}=([\d.e+]+)", s)
            if mm:
                r[f"{k}_ms"] = float(mm[1]) / 1e3
    m = re.search(r"Overall peak memory footprint \(MB\) via periodic monitoring: ([\d.]+)", out)
    if m:
        r["peak_mem_mb"] = float(m[1])
    m = re.search(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \((\w+)\) node, yielding (\d+) partitions", out)
    if m:
        r.update(delegated_nodes=int(m[1]), total_nodes=int(m[2]), delegate=m[3], partitions=int(m[4]))
    if not r.get("avg_ms"):
        err = [l for l in out.splitlines() if re.search(r"error|fail|unsupported", l, re.I)]
        r["error"] = " | ".join(err[-3:])[:400] or out[-400:]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=["fp32", "fp16", "int8_wo", "int8_dyn", "int8_static"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()

    serial = adb("get-serialno").strip()
    model_name = adb("shell", "getprop", "ro.product.model").strip()
    adb("shell", "mkdir", "-p", REMOTE)
    adb("push", BENCH_BIN, REMOTE + "/")
    adb("shell", "chmod", "+x", REMOTE + "/benchmark_model")
    adb("shell", "svc", "power", "stayon", "usb")

    out_dir = HERE / "device_bench"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"{model_name}_{serial}.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["model"], r["variant"], r["config"], r["repeat"]) for r in csv.DictReader(f)}
    fields = ["device", "model", "variant", "config", "repeat", "size_mb", "temp_before", "temp_after", "init_ms",
              "first_ms", "warmup_avg_ms", "avg_ms", "median_ms", "p95_ms", "std_ms", "min_ms", "max_ms",
              "peak_mem_mb", "delegate", "delegated_nodes", "total_nodes", "partitions", "error"]
    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        models = args.models or sorted(p.name for p in (HERE / "tflite" / "arch").iterdir() if p.is_dir())
        for model in models:
            for variant in args.variants:
                local = HERE / "tflite" / "arch" / model / f"{variant}.tflite"
                if not local.exists():
                    continue
                remote = f"{REMOTE}/{model}_{variant}.tflite"
                adb("push", str(local), remote)
                for config in args.configs:
                    for rep in range(REPEATS):
                        if (model, variant, config, str(rep)) in done:
                            continue
                        time.sleep(COOLDOWN)
                        t0 = battery_temp()
                        cmd = (f"cd {REMOTE} && ./benchmark_model --graph={remote} {CONFIGS[config]} --warmup_runs=10 "
                               f"--num_runs=50 --report_peak_memory_footprint=true 2>&1")
                        try:
                            out = adb("shell", cmd)
                        except subprocess.TimeoutExpired:
                            out = "error: timeout"
                        row = dict(device=model_name, model=model, variant=variant, config=config, repeat=rep,
                                   size_mb=round(local.stat().st_size / 2**20, 2), temp_before=t0,
                                   temp_after=battery_temp(), **parse(out))
                        w.writerow(row); f.flush()
                        print(model, variant, config, rep, row.get("median_ms"), row.get("peak_mem_mb"),
                              row.get("error", ""), flush=True)
                        if row.get("error"):
                            break  # mesma falha se repetiria
                adb("shell", "rm", "-f", remote)


if __name__ == "__main__":
    main()
