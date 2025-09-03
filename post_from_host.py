# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess, sys
from inference_request import inference
from launch import lauch_service
import json
import os
app = FastAPI()

class Task(BaseModel):
    task_id: int
    description: str
    inference: list = None
    docker: str = None

run_dict = {"install services": [sys.executable, "launch.py"],
            "inference" : [sys.executable, "inference_request.py"]
            }

os.makedirs("services", exist_ok=True)
@app.post("/task_from_host")
async def task_from_host(task: Task):
    print(f"Received task: {task.task_id}, description: {task.description}")

    try:
        if task.description == "install services":
            list_active_service = lauch_service()
            json.dump(list_active_service, open(f"services/active_services_in_docker_{task.docker}-th.json", "w"), indent=2)
            for s in list_active_service:
                queue 
        elif task.description == "inference":
            information = task.inference
            id_picture = int(task.inference[1])
            model = task.inference[3] 
            # cmd = run_dict[task.description] + information
            dt, result = await inference(id_picture, model)
            # print(f"Running command: {cmd}")
            # subprocess.Popen(cmd)
            print("Started inference request...")
            return {"status": "success", "task_id": task.task_id, "latency_ms": dt, "result": result}
        return {"status": "success", "task_id": task.task_id}

    except Exception as e:
        print(f"Error while processing task: {e}")
        return {"status": "error", "message": str(e)}
