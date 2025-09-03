import os, io, time, argparse
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
import uvicorn

torch.set_num_threads(1)

app = FastAPI()


def load_model(name: str):
    name = name.lower()
    print(f"Loading model: {name}")
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]

    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]

    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]

    elif name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    elif name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    # ---- Detection models ----
    elif name == "ssd":
        weights = models.detection.SSD300_VGG16_Weights.DEFAULT
        model = models.detection.ssd300_vgg16(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    elif name == "retinanet":
        weights = models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.retinanet_resnet50_fpn(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    elif name == "fasterrcnn":
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    elif name == "maskrcnn":
        weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.maskrcnn_resnet50_fpn(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model, preprocess, classes

MODEL_NAME = os.environ.get("MODEL_NAME", "resnet18").lower()
model, preprocess, classes = load_model(MODEL_NAME)

@torch.inference_mode()
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    if MODEL_NAME in ("resnet18", "resnet50"):
        y = model(x)
        probs = torch.softmax(y[0], dim=0)
        k = min(5, probs.numel())
        topk = torch.topk(probs, k=k)
        out = [
            {"class": classes[idx], "prob": float(prob)}
            for prob, idx in zip(topk.values.tolist(), topk.indices.tolist())
        ]
    else:  # SSD
        preds = model(x)[0]
        out = []
        for score, label, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
            s = float(score)
            if s < 0.5:
                continue
            out.append({
                "label": classes[int(label)],
                "score": s,
                "box": [float(v) for v in box.tolist()],
            })

    ms = (time.perf_counter() - t0) * 1000
    return JSONResponse({"model": MODEL_NAME, "ms": ms, "result": out})

@app.get("/health")
async def health():
    return {"model": MODEL_NAME, "status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "resnet18"))
    args = parser.parse_args()

    os.environ["MODEL_NAME"] = args.model
    model, preprocess, classes = load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)