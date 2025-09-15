import asyncio, argparse
import httpx
import numpy as np
import re, sys, subprocess
from env import OffloadingEnv
from ddqn_agent import DDQNAgent
import matplotlib.pyplot as plt
import time
from utils import get_active_service, get_feature
import os
import torch
from trainer_processing_time import DelayPredictor
import pandas as pd
N_SERVER = 4
LGOBAL_SEED = 45

df = {"id": [], "id_picture":[], "predict_cost": [] }
rng = np.random.default_rng(LGOBAL_SEED)

#drl,  random, estimated_process
save_dir = "drl"

def run(user=10, lamd=1.1, host="", port_base=10000, docker_min_max=[], duration=1000):
    list_docker, list_service_in_docker = get_active_service()
    system_arrival_rate = user * lamd
    system_inter_arrival_rate = 1 / system_arrival_rate
    cnt = 0
    agent = DDQNAgent(28, N_SERVER)
    # agent.load()

    # vì OffloadingEnv có async nên ta tạo 1 loop riêng trong thread
    env = asyncio.run(OffloadingEnv(num_servers=4).ainit()) if hasattr(OffloadingEnv, "ainit") else OffloadingEnv(num_servers=3)
    # env.ainit() 
    obs = asyncio.run(env.get_observation())
    queue = {}
    rewards = []
    all_reward = {}

    done = False

    save_dir = "train_result"
    os.makedirs(save_dir, exist_ok=True)
    fearture_vecs = get_feature(obs, id_picture=0, model=0, docker=1) #just for get size
    load_model_estimate_processing_time = DelayPredictor(input_dim=len(fearture_vecs))
    load_model_estimate_processing_time.load_model(f"{save_dir}/pretrained_processing_estimation.pth")
    check_done  = 0
    while duration > 0:
        event = rng.exponential(system_inter_arrival_rate)
        time_sleep = event if event > 0 else 0
        # ngủ sync
        time.sleep(time_sleep)

        # chọn docker
        slected_docker = rng.integers(docker_min_max[0], docker_min_max[1])
        # slected_docker = agent.act(obs) + 1
        # if not (docker_min_max[0] <= slected_docker < docker_min_max[1]):
        #     raise ValueError("action is out of the selected range")
        # chon model
        
        id_picture = rng.integers(0, len(os.listdir("val2017")))

        model = rng.choice(list_service_in_docker[1])
        # model_id = 0
        # if model =="ssd":
        #     model_id = 9
        # start_time = time.perf_counter()
        # def select_server(obs):
        #     slected_docker = 0            
        #     min_processing_predicted_time = float('inf')

        #     for i in range(N_SERVER):
        #         fearture_vecs = np.array(get_feature(obs, id_picture, model_id, i+1)) #id docker from 1
        #         fearture_vecs = np.reshape(fearture_vecs, (1, -1))
        #         fearture_vecs = torch.from_numpy(fearture_vecs).float()
        #         processing_predicted_time = load_model_estimate_processing_time(fearture_vecs)
        #         if min_processing_predicted_time > processing_predicted_time:
        #             slected_docker = i+1
        #             min_processing_predicted_time = processing_predicted_time
        #     return slected_docker
        # predict_cost = time.perf_counter() - start_time
        # df["id"].append(cnt)
        # df["id_picture"].append(id_picture)
        # df['predict_cost'].append(predict_cost)

        # slected_docker = select_server(obs)

        slected_port = port_base + slected_docker


        cmd = [
            sys.executable,
            "host_send_request.py",
            "--request", "inference",
            "--num", "1",
            "--docker", str(slected_docker),
            "--port", str(slected_port),
            "--id", str(cnt),
            "--state", f"{obs}", #save to train predict processing time
            "--model", str(model),
            "--id_picture", str(id_picture),
        ]
        subprocess.Popen(cmd)

        # gọi các async trong thread
        next_state = asyncio.run(env.get_observation())
        # next_state = obs
        queue[cnt] = [obs.copy(), slected_docker - 1, next_state]
        obs = next_state
        reward = asyncio.run(env.get_reward())
        all_reward.update(reward)

        #====================
        def process_rewards():
            nonlocal rewards, all_reward, queue, cnt, done
            del_r_key = []
            asyncio.run(env.get_observation())
            print("env.is_all_queue_end() ---------------> ",env.is_all_queue_end())
            if done and env.is_all_queue_end():
                done= False
            for taskid in list(all_reward.keys()):
                try:
                    re_val = all_reward[taskid][2]
                except Exception as e:
                    # print(f"khong tim thay taskid {taskid} {e}")
                    continue
                try:
                    if re_val != "None" and re_val is not None:
                        train_data = queue[int(taskid)]
                        train_data.append(-float(re_val))
                        if re_val > 10:
                            done = True
                            agent.remember(train_data[0], train_data[1], train_data[2], train_data[3], True)
                        else:
                            agent.remember(train_data[0], train_data[1], train_data[2], train_data[3], False)
              
                        rewards.append(-float(re_val))
                        del queue[int(taskid)]
                        del_r_key.append(taskid)  
                        print(f"---------> {re_val}")
                
                except Exception as e:
                    print(f"Chua tim thay {taskid} {e}")
                    continue
                
            for k in del_r_key:
                del all_reward[k]
            agent.update()
        asyncio.run(asyncio.to_thread(process_rewards))
        #====================
        # if check_done >100:
        #     done = True
        #     check_done=0
        while done:
            print ("dang chay o day -----")
            #sau khi done thì vẫn còn các nhiệm vụ trong queue, phải chờ cho chúng kết thúc rồi ms chuyển qua epoch mới
            asyncio.run(asyncio.to_thread(process_rewards))
        done = False
        if cnt > 0 and cnt % 100 == 0:
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward')
            plt.savefig('rewards.png')
            plt.close()
            agent.save()
        check_done += 1

        duration -= event
        cnt += 1



if __name__ == "__main__":
    run(15, 1, "", 10000, [1, 5], 300)
    df = pd.DataFrame(df)
    df.to_csv("cost.csv", index=None)

