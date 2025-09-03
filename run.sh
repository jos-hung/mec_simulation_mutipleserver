#!/bin/bash

python scripts/loadgen.py --model resnet18 --total 100 --concurrency 64 --scheduler http://127.0.0.1:8510 &
python scripts/loadgen.py --model resnet50 --total 100 --concurrency 64 &
python scripts/loadgen.py --model ssd --total 100 --concurrency 64 &
python scripts/loadgen.py --model resnet101 --total 100 --concurrency 64 &
python scripts/loadgen.py --model resnet34 --total 100 --concurrency 64 &
python scripts/loadgen.py --model efficientnet_b0 --total 100 --concurrency 64 &
python scripts/loadgen.py --model mobilenetv2 --total 100 --concurrency 64 &
python scripts/loadgen.py --model retinanet --total 100 --concurrency 64 &
python scripts/loadgen.py --model fasterrcnn --total 100 --concurrency 64 &
python scripts/loadgen.py --model maskrcnn --total 100 --concurrency 64 &
wait
echo "All loadgen finished"
