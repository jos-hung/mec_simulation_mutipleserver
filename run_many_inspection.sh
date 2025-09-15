.venv/bin/python3 display_computing_result.py
#experiment_types = ["random", "drl_train", "drl_prediction", "esimated_processing_time"]
.venv/bin/python3 trainer.py --n_users 15 --lamd 1 --docker_min_max 1 5 --duration 1000 --output_file output_file --experiment_type 0 --LGOBAL_SEED 45
.venv/bin/python3 trainer_processing_time.py
.venv/bin/python3 trainer.py --n_users 15 --lamd 1 --docker_min_max 1 5 --duration 300 --output_file output_file --experiment_type 3 --LGOBAL_SEED 42
.venv/bin/python3 trainer.py --n_users 15 --lamd 1 --docker_min_max 1 5 --duration 1000 --output_file output_file --experiment_type 1 --LGOBAL_SEED 45
.venv/bin/python3 trainer.py --n_users 15 --lamd 1 --docker_min_max 1 5 --duration 300 --output_file output_file --experiment_type 2 --LGOBAL_SEED 45
