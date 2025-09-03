# MEC Process Microservices Pack (No Docker) — Đúng yêu cầu

Bộ này triển khai đúng yêu cầu: **“server ảo” bằng *processes***, mỗi model AI là **một microservice** chạy trên **cổng riêng**, có **scheduler** (process) phân phối request tới các server ảo bằng **round‑robin**, và **load generator** để bắn tới **1e5 request** đo *throughput/latency*.

> Mặc định chạy CPU. Nếu có GPU, bạn có thể chỉnh sang bản CUDA (ghi chú ở cuối).

---

## 0) Cấu trúc thư mục
```
edge-mec/
├─ requirements.txt
├─ config.yaml                 # cấu hình cổng cho các "server ảo" và scheduler
├─ launch.py                   # spawn các process dịch vụ + scheduler
├─ kill_all.py                 # dừng toàn bộ process theo pids.json
├─ pids.json                   # file PID (tự tạo khi launch)
├─ infer_service_proc.py       # microservice ResNet18/50, SSD
├─ scheduler_proc.py           # scheduler phân phối + metrics
└─ scripts/
   └─ loadgen.py               # load generator gửi tới scheduler (1e5 req)
```

---

## 1) requirements.txt
```text
fastapi
uvicorn[standard]
httpx
numpy
pillow
pyyaml
psutil
torch==2.2.*
torchvision==0.17.*
```

---

## 2) config.yaml (ví dụ 2 replica/Model)
```yaml
services:
  resnet18: ["127.0.0.1:8101", "127.0.0.1:8102"]
  resnet50: ["127.0.0.1:8111", "127.0.0.1:8112"]
  ssd:      ["127.0.0.1:8121", "127.0.0.1:8122"]

scheduler:
  host: 0.0.0.0
  port: 7000
```

---

## 3) infer_service_proc.py — microservice theo *process*
```python
import os, io, time, argparse
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
import uvicorn

# Giảm tranh chấp CPU khi nhiều process
torch.set_num_threads(1)

app = FastAPI()


def load_model(name: str):
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights).eval()
        preprocess = weights.transforms()
        classes = weights.meta["categories"]
    elif name == "ssd":
        weights = models.detection.SSD300_VGG16_Weights.DEFAULT
        model = models.detection.ssd300_vgg16(weights=weights).eval()
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
```

---

## 4) scheduler_proc.py — scheduler + metrics
```python
import os, time, itertools, statistics
import httpx, yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
POOLS = {k.lower(): [f"http://{u}" for u in v] for k, v in cfg.get("services", {}).items()}
ITER = {k: itertools.cycle(v) for k, v in POOLS.items() if v}

METRICS = {"count": 0, "latencies": [], "start": time.time()}

@app.post("/infer/{model}")
async def route(model: str, file: UploadFile = File(...)):
    model = model.lower()
    if model not in POOLS or not POOLS[model]:
        return JSONResponse({"error": f"No backends for {model}"}, status_code=503)

    url = next(ITER[model]) + "/infer"
    data = await file.read()
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, files={"file": (file.filename, data, file.content_type or "application/octet-stream")})
    dt = (time.perf_counter() - t0) * 1000

    METRICS["latencies"].append(dt)
    METRICS["count"] += 1

    return JSONResponse({"backend": url, "ms": dt, "payload": r.json()})

@app.get("/metrics")
async def metrics():
    dur = max(1e-6, time.time() - METRICS["start"])
    count = METRICS["count"]
    tput = count / dur
    lat = METRICS["latencies"]
    if lat:
        avg = sum(lat)/len(lat)
        p50 = statistics.median(lat)
        p95 = statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat)
        p99 = statistics.quantiles(lat, n=100)[98] if len(lat) >= 100 else max(lat)
    else:
        avg = p50 = p95 = p99 = 0.0
    return {
        "requests": count,
        "throughput_rps": tput,
        "latency_ms": {"avg": avg, "p50": p50, "p95": p95, "p99": p99},
        "backends": {k: len(v) for k, v in POOLS.items()},
    }

@app.get("/health")
async def health():
    return {"status": "ok", "backends": {k: len(v) for k, v in POOLS.items()}}

if __name__ == "__main__":
    host = cfg.get("scheduler", {}).get("host", "0.0.0.0")
    port = int(cfg.get("scheduler", {}).get("port", 7000))
    uvicorn.run(app, host=host, port=port)
```

---

## 5) launch.py — tạo "server ảo" bằng *processes*
```python
import subprocess, sys, json, time, yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

pids = {"services": [], "scheduler": None}

# Launch services
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
```

---

## 6) kill_all.py — dừng toàn bộ process
```python
import json, os, signal, time

try:
    with open("pids.json", "r") as f:
        pids = json.load(f)
except FileNotFoundError:
    print("No pids.json found.")
    raise SystemExit(0)

all_pids = []
if pids.get("scheduler"):
    all_pids.append(pids["scheduler"])
all_pids += pids.get("services", [])

for pid in all_pids:
    try:
        os.kill(pid, signal.SIGTERM)
        print("SIGTERM", pid)
    except ProcessLookupError:
        pass

time.sleep(1)
for pid in all_pids:
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
        print("SIGKILL", pid)
    except ProcessLookupError:
        pass
print("All processes terminated.")
```

---

## 7) scripts/loadgen.py — tạo tới 1e5 request
```python
import argparse, asyncio, glob, os, random, statistics, time
import httpx

async def worker(client, url, files, total, results):
    i = 0
    n = len(files)
    while True:
        if total[0] >= total[1]:
            return
        path = files[i % n]
        i += 1
        t0 = time.perf_counter()
        try:
            with open(path, "rb") as f:
                r = await client.post(url, files={"file": (os.path.basename(path), f, "application/octet-stream")})
            r.raise_for_status()
            dt = (time.perf_counter() - t0) * 1000
            results.append(dt)
        except Exception:
            pass
        total[0] += 1

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", default="http://127.0.0.1:7000")
    ap.add_argument("--model", choices=["resnet18","resnet50","ssd"], default="resnet18")
    ap.add_argument("--images", default="./images")
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    url = f"{args.scheduler}/infer/{args.model}"
    os.makedirs(args.images, exist_ok=True)

    # Nếu thư mục trống, tạo một vài ảnh synthetic nhẹ để thử tải
    imgs = glob.glob(os.path.join(args.images, "*"))
    if not imgs:
        from PIL import Image
        for i in range(8):
            img = Image.new("RGB", (224,224), (i*30 % 255, i*60 % 255, i*90 % 255))
            img.save(os.path.join(args.images, f"synthetic_{i}.png"))
        imgs = glob.glob(os.path.join(args.images, "*"))

    random.shuffle(imgs)

    results = []
    total = [0, args.total]
    t0 = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [worker(client, url, imgs, total, results) for _ in range(args.concurrency)]
        await asyncio.gather(*tasks)

    dur = time.time() - t0
    rps = args.total / dur if dur > 0 else 0
    if results:
        avg = sum(results)/len(results)
        p50 = statistics.median(results)
        p95 = statistics.quantiles(results, n=20)[18] if len(results) >= 20 else max(results)
        p99 = statistics.quantiles(results, n=100)[98] if len(results) >= 100 else max(results)
    else:
        avg = p50 = p95 = p99 = 0.0

    print({
        "sent": args.total,
        "duration_s": dur,
        "throughput_rps": rps,
        "latency_ms": {"avg": avg, "p50": p50, "p95": p95, "p99": p99},
    })

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8) Cách chạy nhanh (PROCESS MODE)
```bash
# 0) Python >= 3.10, venv (khuyến nghị)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Sửa config.yaml nếu muốn thay đổi cổng/replica

# 2) Launch các process (server ảo + scheduler)
python launch.py
# → Scheduler mặc định tại http://127.0.0.1:7000

# 3) Bắn tải benchmark (ví dụ 100k request, 64 luồng, model ResNet18)
python scripts/loadgen.py --model resnet18 --total 100000 --concurrency 64

# 4) Xem metrics realtime của scheduler
curl http://127.0.0.1:7000/metrics | jq

# 5) Dừng toàn bộ process
python kill_all.py
```

---

## 9) Gợi ý tối ưu & mở rộng
- **Affinity/Isolates CPU**: chạy mỗi replica với `taskset`/`numactl` để cô lập core ⇒ đo được scaling sạch hơn.
- **p95/p99 ổn định hơn**: tăng thời gian chạy + số request, tránh warming bias (model lần đầu load nặng).
- **GPU**: nếu có CUDA, thay PyTorch bản CUDA, và trong `launch.py` bạn có thể gán `CUDA_VISIBLE_DEVICES` khác nhau cho từng replica.
- **Bảo trì**: chuyển metrics sang Prometheus/Grafana sau khi xác nhận pipeline ổn.

---

## 10) Ghi chú GPU (tuỳ chọn)
- Cài torch/torchvision bản CUDA tương ứng (ví dụ CUDA 12.1). Ví dụ: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- Thêm biến môi trường khi spawn:
  - Trong `launch.py` trước `subprocess.Popen`, set `env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "0"` (hoặc "1" …), rồi `Popen(cmd, env=env)`.

---

## 11) (Tuỳ chọn) Bản Docker
Nếu muốn dùng Docker thay vì processes, dùng bản Compose ở tài liệu trước. Nhưng theo yêu cầu này, **PROCESS MODE** là mặc định.

