import asyncio
from typing import Dict
from fastapi import FastAPI, Request
import uvicorn
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import os, io, time, argparse
from torchvision import models
import torch

class Service:
    def __init__(self, name: str):
        self.name = name
        self.queue = asyncio.Queue()
        self.app = FastAPI()
        self._service_inference()
    # async def run(self):
    #     while True:
    #         task = await self.queue.get()
    #         print(f"{self.name} processing {task}")
    #         # Simulate inference latency
    #         await asyncio.sleep(0.01)
    #         self.queue.task_done()
    
    def load_model(self, name: str):
        name = name.lower()

        # ---- Classification models ----
        if self.name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
            self.model = models.resnet34(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT
            self.model = models.resnet101(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "mobilenetv2":
            weights = models.MobileNet_V2_Weights.DEFAULT
            self.model = models.mobilenet_v2(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        # ---- Detection models ----
        elif self.name == "ssd":
            weights = models.detection.SSD300_VGG16_Weights.DEFAULT
            self.model = models.detection.ssd300_vgg16(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "retinanet":
            weights = models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
            self.model = models.detection.retinanet_resnet50_fpn(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "fasterrcnn":
            weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = models.detection.fasterrcnn_resnet50_fpn(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        elif self.name == "maskrcnn":
            weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = models.detection.maskrcnn_resnet50_fpn(weights=weights).eval()
            self.preprocess = weights.transforms()
            self.classes = weights.meta["categories"]

        else:
            raise ValueError(f"Unsupported model: {name}")

            
    def _service_inference(self):
        @self.app.post("/infer")
        async def infer(file: UploadFile = File(...)):
            t0 = time.perf_counter()
            data = await file.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            x = self.preprocess(img).unsqueeze(0)

            out = None
            # ---- Classification models ----
            if self.name in (
                "resnet18", "resnet50",
                "mobilenetv2", "efficientnet_b0",
                "resnet34", "resnet101"
            ):
                y = self.model(x)
                probs = torch.softmax(y[0], dim=0)
                k = min(5, probs.numel())
                topk = torch.topk(probs, k=k)
                out = [
                    {"class": self.classes[idx], "prob": float(prob)}
                    for prob, idx in zip(topk.values.tolist(), topk.indices.tolist())
                ]

            # ---- Detection models ----
            elif self.name in ("ssd", "retinanet", "fasterrcnn", "maskrcnn"):
                preds = self.model(x)[0]
                out = []
                for score, label, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
                    if float(score) < 0.5:
                        continue
                    result = {
                        "label": self.classes[int(label)],
                        "score": float(score),
                        "box": [float(v) for v in box.tolist()],
                    }
                    # Mask R-CNN có thêm mask
                    if "masks" in preds:
                        mask = preds["masks"][0, 0].detach().cpu().numpy().tolist()
                        result["mask"] = mask
                    out.append(result)

            else:
                raise ValueError(f"Unsupported model: {self.name}")

            ms = (time.perf_counter() - t0) * 1000
            return JSONResponse({"model": self.name, "ms": ms, "result": out})

class Server:
    def __init__(self, name: str, port: int, host: str = "0.0.0.0"):
        self.name = name
        self.port = port
        self.host = host
        self.services: Dict[str, Service] = {}
        self.local_scheduler_queue = asyncio.Queue()
        self.app = FastAPI()
        self._setup_routes()

    def add_service(self, service: Service):
        self.services[service.name] = service

    def _setup_routes(self):
        @self.app.post("/schedule/{model_name}")
        async def schedule(model_name: str, request: Request):
            data = await request.json()
            # Put task into local scheduler
            await self.local_scheduler_queue.put((model_name, data))
            return {"status": "queued", "model": model_name, "task": data}
        
        @self.app.get("/add_service")
        async def add_service_endpoint(name: str, cores: str):
            core_list = [int(c) for c in cores.split(",")]
            if name in self.services:
                return {"status": "error", "message": "Service already exists"}
            self.add_service(Service(name, core_list))
            return {"status": "ok", "service": name, "assigned_cores": core_list}
        

    async def local_scheduler(self):
        while True:
            model_name, task = await self.local_scheduler_queue.get()
            if model_name in self.services:
                await self.services[model_name].queue.put(task)
            else:
                print(f"Service {model_name} not found")
            self.local_scheduler_queue.task_done()

    async def run_services(self):
        tasks = [asyncio.create_task(s.run()) for s in self.services.values()]
        tasks.append(asyncio.create_task(self.local_scheduler()))
        await asyncio.gather(*tasks)

    def start(self):
        asyncio.run(self._main())

    async def _main(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await asyncio.sleep(0.01) # Give server time to start
        print(f"Server {self.name} running on {self.host}:{self.port}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Server_1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    server = Server(name = args.name,host=args.host, port=args.port)
    server.start()
