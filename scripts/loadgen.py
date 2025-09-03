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
    ap.add_argument("--scheduler", default="http://0.0.0.0:8000")
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