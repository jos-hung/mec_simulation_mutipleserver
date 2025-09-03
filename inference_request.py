
import argparse, asyncio, glob, os, random, statistics, time
import httpx
import yaml
from PIL import Image
import io
SHARED_PATH = "/shared" 

async def inference(id_picture:int, model:str):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg.get("scheduler").get("host", "")
    host = cfg.get('scheduler').get("host", "")
    port = cfg.get('scheduler').get("port", "")
    url = f"http://{host}:{port}/infer/{model}"
    print(id_picture)
    picture = os.listdir(SHARED_PATH)[int(id_picture)]
    path = os.path.join(SHARED_PATH, picture)
    t0 = time.perf_counter()
    
    img = Image.open(path)
    img = img.resize((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0) 
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, files={"file": (os.path.basename(path), buf, "application/octet-stream")})
            r.raise_for_status()
        dt = (time.perf_counter() - t0) * 1000
        return dt, r.json()
    except Exception as e:
        print(f"error {e}")
        return None, None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id_picture", type=int, required=True)
    ap.add_argument("--model", type=str, default="resnet18")
    args = ap.parse_args()
    asyncio.run(inference(args.id_picture, args.model))
    