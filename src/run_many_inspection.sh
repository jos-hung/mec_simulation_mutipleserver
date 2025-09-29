#!/bin/bash

num_users=(10 15 20 25 30)

for item in "${num_users[@]}"; do
    echo "Running simulation with $item users"
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item --lamd 1 --docker_min_max 1 5 --duration 1500 --output_file output_file --experiment_type 0 --LGOBAL_SEED 45

    sleep 20
    ../.venv/bin/python3 trainer_processing_time.py --n_users $item --seed 45

    #estimate the processing time
    #=============================
    sleep 20
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 3 --LGOBAL_SEED 42
    sleep 20
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 3 --LGOBAL_SEED 50
    sleep 20
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 3 --LGOBAL_SEED 55
    #=============================
    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item --lamd 1 --docker_min_max 1 5 --duration 1500 --output_file output_file --experiment_type 1 --LGOBAL_SEED 45

    #=============================
    #eval drl method
    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item   --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 2 --LGOBAL_SEED 42
    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item   --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 2 --LGOBAL_SEED 50
    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item   --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 2 --LGOBAL_SEED 55
    #=============================

    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item --lamd 1 --docker_min_max 1 5 --duration 1500 --output_file output_file --experiment_type 4 --LGOBAL_SEED 45
    #=============================
    #drl with history task information
    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item   --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 5 --LGOBAL_SEED 42
    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item   --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 5 --LGOBAL_SEED 50
    sleep 50
    source clearqueue.sh
    ../.venv/bin/python3 trainer.py --n_users $item   --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 5 --LGOBAL_SEED 55
    #=============================
done
