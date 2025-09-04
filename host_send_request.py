import asyncio, argparse
import httpx
import numpy as np
import os
import yaml
import json
import re


service_dir = "service"
list_service_in_docker = {}

for filename in os.listdir(service_dir):
    match = re.match(r"active_services_in_docker_(\d+)-th\.json", filename)
    if match:
        docker_id = int(match.group(1))
        filepath = os.path.join(service_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
            list_service_in_docker[docker_id] = list(data.keys())

list_docker = list(list_service_in_docker.keys())

async def send_tasks(task_num, url, request = "", docker = None):
    cnt=0
    while task_num:
        inter_arrival_time = np.random.exponential(1)
        print(f"Sleeping for {inter_arrival_time} seconds")
        await asyncio.sleep(inter_arrival_time)  # <-- async sleep

        id_picture = np.random.randint(0, len(os.listdir("val2017")))
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        model = "None"
        if docker is not None:
            docker = np.random.choice(list_docker)
            model = np.random.choice(list_service_in_docker[docker])
        payload = {
            "task_id": cnt,
            "description": f"{request}",
            "inference": ["--id_picture", str(id_picture), "--model", model],
            "docker": str(1)
        }
        print(f"Sending task with id_picture={id_picture}, model={model}")
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, json=payload)  
            print(response.json())
        task_num -= 1
        cnt += 1
if __name__ == "__main__":
    import asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=str, required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--docker", type=str,default=None)
    # ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=str, default="10000")
    args = ap.parse_args()
    port = args.port
    # asyncio.run(send_tasks(task_num=args.num, url=f"http://localhost:{args.port}/handle_host_request", request=args.request, docker = args.docker))
    asyncio.run(send_tasks(task_num=1, url=f"http://localhost:{port}/handle_host_request", request=args.request, docker = args.docker))
