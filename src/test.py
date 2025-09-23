import asyncio, argparse
import httpx
import numpy as np
import os
import yaml
import json
import re, time, sys
import subprocess




def run(user=10, lamd = 1.1, host = "", port_base = 10000,docker_min_max =[], duration = 1000):
    system_arrival_rate = user*lamd
    system_inter_arrival_rate = 1/system_arrival_rate
    time_start = 0
    cnt = 0
    while duration > 0:
        event = np.random.exponential(system_inter_arrival_rate)
        slected_docker = np.random.randint(docker_min_max[0], docker_min_max[1])
        slected_port = port_base + slected_docker
        time.sleep(event)
        cmd = [
            sys.executable,
            "host_send_request.py",
            "--request", "inference",
            "--num", "1",
            "--docker", str(slected_docker),
            "--port", str(slected_port),
            "--id", str(cnt)
        ]

        proc = subprocess.Popen(cmd)
        duration -= event
        cnt+=1
        
run(40, 1.1, "", 10000, [1, 4], 100)