#!/bin/bash
# source clearqueue.sh
# ../.venv/bin/python3 trainer.py --n_users 10 --lamd 1 --docker_min_max 1 5 --duration 1500 --output_file output_file --experiment_type 0 --LGOBAL_SEED 45
# sleep 20

# ../.venv/bin/python3 trainer_processing_time.py
# sleep 2
# source clearqueue.sh

# ../.venv/bin/python3 trainer.py --n_users 10 --lamd 1 --docker_min_max 1 5 --duration 400 --output_file output_file --experiment_type 3 --LGOBAL_SEED 42
# sleep 50
# source clearqueue.sh

../.venv/bin/python3 trainer.py --n_users 10 --lamd 1 --docker_min_max 1 5 --duration 300 --output_file output_file --experiment_type 4 --LGOBAL_SEED 45
sleep 100
source clearqueue.sh

../.venv/bin/python3 trainer.py --n_users 10   --lamd 1 --docker_min_max 1 5 --duration 100 --output_file output_file --experiment_type 4 --LGOBAL_SEED 42
