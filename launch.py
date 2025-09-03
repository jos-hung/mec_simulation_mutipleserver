import subprocess, sys, json, time, yaml
import os
import time
import numpy as np
import platform
import psutil
import threading


def lauch_service():
    os.system("clear")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    pids = {"services": [], "scheduler": None}

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    # Launch scheduler (đọc config.yaml)
    proc = subprocess.Popen([sys.executable, "scheduler_proc.py"])
    pids["scheduler"] = proc.pid
    print(f"Scheduler started (pid={proc.pid})")
    pros = []

    keys = list(cfg.get("services", {}).keys())

    #because if index of serivce high then it will require more time for inference
    #so we will set it with low probability for random choice
    prob = np.array([1/(i+1) for i in range(len(keys))])
    prob = prob/prob.sum()

    list_service = {}
    for i in range(cfg.get("max_service", 5)):
        model = np.random.choice(range(len(keys)), p=prob)
        if (i <=2) and model>6:
            list_service[model] = cfg.get("services", {}).get(keys[model], [])
            break
        if i>2 and model>6:
            continue
        list_service[keys[model]] = cfg.get("services", {}).get(keys[model], [])

    print("Services to launch:", list_service)

    for model, endpoints in list_service.items():
        for ep in endpoints:
            host, port = ep.split(":")
            cmd = [sys.executable, "infer_service_proc.py", "--model", model, "--host", host, "--port", port]
            with open(f"{log_dir}/{model}_log.txt", "w") as log_file:
                proc = subprocess.Popen(cmd, stderr=log_file)
                pros.append(proc)
            pids["services"].append(proc.pid)
            print(f"Started {model} at {ep} (pid={proc.pid})")
            time.sleep(2)  # stagger


    with open("pids.json", "w") as f:
        json.dump(pids, f, indent=2)
    print("PIDs saved to pids.json. Dừng bằng: python kill_all.py")
    
    return list_service