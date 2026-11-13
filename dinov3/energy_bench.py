"""Energia por inferência no aparelho Android, fora da tomada (adb por Wi-Fi).

Protocolo por modelo x variante (XNNPACK, 4 threads):
  1. resfriamento de COOLDOWN s;
  2. repouso: 30 s com a mesma tela (brilho mínimo, ligada) -> P_idle;
  3. carga: benchmark_model com --min_secs=60 -> P_load e latência média;
  4. E_inf = (P_load - P_idle) * latência média  (energia atribuível ao modelo, J)
     E_total = P_load * latência média           (energia total do aparelho por inferência, J)
Corrente e tensão lidas de /sys/class/power_supply/battery/{current_now,voltage_now}
por um laço no próprio aparelho (~5 Hz), com o tempo de /proc/uptime.
Uso: python energy_bench.py --serial <ip:porta> [--repeats 3] [--idle 60] [--tag _v2]
v2 (07/11): registra a frequência do núcleo grande (cpu4) para documentar limitação térmica/bateria.
Saída: device_bench/energy_<modelo>_<serial>.csv
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
PS = "/sys/class/power_supply/battery"
COOLDOWN = 30
IDLE_S = 30
LOAD_S = 60
PLAN = [("mobilenet_v2", "fp32"), ("dinov3_small", "fp32"), ("dinov3_small", "int8_wo"),
        ("dinov3_small", "int8_dyn"), ("dinov3_base", "int8_wo"), ("dinov3_base", "int8_dyn")]


def adb(serial, *args, timeout=900):
    return subprocess.run([ADB, "-s", serial, *args], capture_output=True, text=True, timeout=timeout).stdout


def keep_awake(serial):
    """Acorda a tela e impede o modo Doze (o S8 entrou em Dozing na primeira execução)."""
    # KEYCODE_SHIFT_LEFT conta como atividade do usuário (reinicia o tempo de tela) sem abrir nada;
    # é enviado só no início de cada repetição, fora das janelas de medição.
    adb(serial, "shell", "input keyevent KEYCODE_WAKEUP; input keyevent KEYCODE_SHIFT_LEFT; svc power stayon true; "
                         "settings put system screen_off_timeout 1800000; dumpsys deviceidle disable >/dev/null")


def wakefulness(serial):
    m = re.search(r"mWakefulness=(\w+)", adb(serial, "shell", "dumpsys power"))
    return m[1] if m else "?"


def uptime(serial):
    return float(adb(serial, "shell", "cat /proc/uptime").split()[0])


def read_samples(serial, path):
    rows = []
    for line in adb(serial, "shell", f"cat {path}").splitlines():
        p = line.split()
        if len(p) >= 3:
            try:
                rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]) if len(p) > 3 else float("nan")))
            except ValueError:
                pass
    return rows


def power_w(rows, t0, t1):
    sel = [(i, v) for t, i, v, _ in rows if t0 <= t <= t1]
    if not sel:
        return None, 0
    # Samsung: corrente em mA (|i| < 20000) ou µA; tensão em µV.
    p = [abs(i) * (1e-3 if abs(i) < 20000 else 1e-6) * v * 1e-6 for i, v in sel]
    return sum(p) / len(p), len(p)


def freq_mhz(rows, t0, t1):
    f = [x for t, _, _, x in rows if t0 <= t <= t1 and x == x]
    return sum(f) / len(f) / 1e3 if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--idle", type=int, default=IDLE_S)
    ap.add_argument("--tag", default="")
    ap.add_argument("--current_node", default="current_now",
                    help="no S8, batt_current_ua_now (µA, ~5 Hz); current_now só atualiza a cada ~10 s")
    args = ap.parse_args()
    s = args.serial
    model_name = adb(s, "shell", "getprop", "ro.product.model").strip()
    status = adb(s, "shell", f"cat {PS}/status").strip()
    assert status.lower().startswith("discharg"), f"aparelho carregando ({status}): desconecte o cabo"
    adb(s, "shell", "mkdir", "-p", REMOTE)
    adb(s, "push", BENCH_BIN, REMOTE + "/")
    adb(s, "shell", "chmod", "+x", REMOTE + "/benchmark_model")
    adb(s, "shell", "settings put system screen_brightness_mode 0; settings put system screen_brightness 0")
    keep_awake(s)

    idle_s = args.idle
    out_csv = HERE / "device_bench" / f"energy{args.tag}_{model_name}_{s.replace(':', '_')}.csv"
    fields = ["device", "model", "variant", "repeat", "awake_before", "awake_after", "battery_pct", "temp_c", "p_idle_w", "p_load_w", "n_idle", "n_load",
              "count", "avg_ms", "f_idle_mhz", "f_load_mhz", "e_model_mj", "e_total_mj", "error"]
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for model, variant in PLAN:
            local = HERE / "tflite" / "arch" / model / f"{variant}.tflite"
            remote = f"{REMOTE}/{model}_{variant}.tflite"
            adb(s, "push", str(local), remote)
            for rep in range(args.repeats):
                time.sleep(COOLDOWN)
                keep_awake(s); time.sleep(2)
                awake_before = wakefulness(s)
                log = f"{REMOTE}/power.log"
                adb(s, "shell", f"rm -f {log}; nohup sh -c 'while true; do echo $(cut -d\" \" -f1 /proc/uptime) "
                                f"$(cat {PS}/{args.current_node}) $(cat {PS}/voltage_now) $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq); sleep 0.2; done' > {log} 2>/dev/null &")
                t_idle0 = uptime(s); time.sleep(idle_s); t_idle1 = uptime(s)
                out = adb(s, "shell", f"cd {REMOTE} && ./benchmark_model --graph={remote} --use_xnnpack=true --num_threads=4 "
                                      f"--warmup_runs=10 --num_runs=50 --min_secs={LOAD_S} 2>&1")
                t_load1 = uptime(s)
                awake_after = wakefulness(s)
                adb(s, "shell", "pkill -f 'sleep 0.2' ; pkill -f power.log; true")
                rows = read_samples(s, log)
                m = re.search(r"Inference \(avg\): ([\d.e+]+)", out)
                c = [l for l in out.splitlines() if "count=" in l]
                avg_ms = float(m[1]) / 1e3 if m else None
                count = int(re.search(r"count=(\d+)", c[-1])[1]) if c else None
                p_idle, n_idle = power_w(rows, t_idle0, t_idle1)
                # Janela de carga: os últimos LOAD_S s (exclui aquecimento e carga do modelo).
                p_load, n_load = power_w(rows, t_load1 - LOAD_S, t_load1)
                ok = None not in (avg_ms, p_idle, p_load) and awake_before == awake_after == "Awake"
                row = dict(device=model_name, model=model, variant=variant, repeat=rep,
                           awake_before=awake_before, awake_after=awake_after,
                           battery_pct=adb(s, "shell", f"cat {PS}/capacity").strip(),
                           temp_c=int(adb(s, "shell", f"cat {PS}/temp").strip() or 0) / 10,
                           p_idle_w=p_idle, p_load_w=p_load, n_idle=n_idle, n_load=n_load, count=count, avg_ms=avg_ms,
                           f_idle_mhz=freq_mhz(rows, t_idle0, t_idle1), f_load_mhz=freq_mhz(rows, t_load1 - LOAD_S, t_load1),
                           e_model_mj=(p_load - p_idle) * avg_ms if ok else None,
                           e_total_mj=p_load * avg_ms if ok else None,
                           error="" if ok else f"estado {awake_before}->{awake_after}; " + out[-250:])
                w.writerow(row); f.flush()
                print(model, variant, rep, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                                            if k in ("p_idle_w", "p_load_w", "avg_ms", "f_load_mhz", "e_model_mj", "n_load", "awake_after")}, flush=True)
            adb(s, "shell", "rm", "-f", remote)
    adb(s, "shell", "svc power stayon false; dumpsys deviceidle enable >/dev/null")


if __name__ == "__main__":
    main()
