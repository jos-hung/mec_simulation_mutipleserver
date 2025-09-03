import asyncio
import httpx
import numpy as np
import os
import yaml

async def send_tasks(task_num, url):
    while task_num:
        inter_arrival_time = np.random.exponential(1)
        print(f"Sleeping for {inter_arrival_time} seconds")
        await asyncio.sleep(inter_arrival_time)  # <-- async sleep

        id_picture = np.random.randint(0, len(os.listdir("val2017")))
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        model = np.random.choice(list(cfg.get("services").keys()))

        payload = {
            "task_id": 1,
            "description": "install services",
            "inference": ["--id_picture", str(id_picture), "--model", model],
            "docker": "1"
        }
        print(f"Sending task with id_picture={id_picture}, model={model}")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)  
            print(response.json())
        exit(0)
        task_num -= 1
# Chạy async
import asyncio
asyncio.run(send_tasks(task_num=10000, url="http://localhost:10005/task_from_host"))
