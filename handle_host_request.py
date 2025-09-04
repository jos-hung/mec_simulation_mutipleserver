# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess, sys
from inference import inference
from launch import lauch_service
import json
import os
import asyncio
app = FastAPI()

class Task(BaseModel):
    task_id: int
    description: str
    inference: list = None
    docker: int

run_dict = {"install": [sys.executable, "launch.py"],
            "inference" : [sys.executable, "inference.py"],
            "kill":[sys.executable, "kill_all.py"]
            }

queue = asyncio.Queue()
results = asyncio.Queue()

SERVICE_PATH = "/service" 
async def worker():
    while True:
        if not queue.empty():
            task: Task = await queue.get()
            try:
                id_picture = int(task.inference[1])
                model = task.inference[3] 
                dt, result = await inference(id_picture, model)
                print(f"[Worker] Finished inference request for task {task.task_id}")
                await results.put([task.task_id, dt, result])
            except Exception as e:
                await results.put([task.task_id, "error", str(e)])
            finally:
                queue.task_done()
        await asyncio.sleep(0.5)

@app.post("/handle_host_request")
async def handle_host_request(task: Task):
    print(f"Received task: {task.task_id}, description: {task.description}")
    try:
        if task.description == "install":
            list_active_service = lauch_service()
            filename = f"{SERVICE_PATH}/active_services_in_docker_{int(task.docker)}-th.json"
            with open(filename, "w") as f:
                json.dump(list_active_service, f, indent=2)
        elif task.description == "inference":
            await queue.put(task)
            cur_result = []
            while not results.empty():
                re: list = await results.get()
                cur_result.append(re)
            return {"status": "queued", "task_id": task.task_id,"current result": cur_result }
        elif task.description == "kill":
            proc = subprocess.Popen(run_dict[task.description ])
        return {"status": "success", "task_id": task.task_id}

    except Exception as e:
        print(f"Error while processing task: {e}")
        return {"status": "error", "message": str(e)}
    
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())
