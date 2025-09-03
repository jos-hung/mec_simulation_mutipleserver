#!/bin/bash

python scripts/loadgen.py --model resnet18 --total 100 --concurrency 64 &
python scripts/loadgen.py --model resnet50 --total 100 --concurrency 64 &
python scripts/loadgen.py --model ssd --total 100 --concurrency 64 &

wait
echo "All loadgen finished"
