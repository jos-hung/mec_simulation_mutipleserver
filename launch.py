import subprocess, sys, json, time, yaml
import os
import time
import numpy as np
import platform
import psutil
import threading

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

pids = {"services": [], "scheduler": None}

for model, endpoints in cfg.get("services", {}).items():
    for ep in endpoints:
        host, port = ep.split(":")
        cmd = [sys.executable, "infer_service_proc.py", "--model", model, "--host", host, "--port", port]
        proc = subprocess.Popen(cmd)
        pids["services"].append(proc.pid)
        print(f"Started {model} at {ep} (pid={proc.pid})")
        time.sleep(0.2)  # stagger

# Launch scheduler (đọc config.yaml)
proc = subprocess.Popen([sys.executable, "scheduler_proc.py"])
pids["scheduler"] = proc.pid
print(f"Scheduler started (pid={proc.pid})")

with open("pids.json", "w") as f:
    json.dump(pids, f, indent=2)
print("PIDs saved to pids.json. Dừng bằng: python kill_all.py")