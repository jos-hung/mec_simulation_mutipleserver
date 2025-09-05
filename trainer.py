import asyncio, argparse
import httpx
import numpy as np
import os
import yaml
import json
import re, sys, subprocess
from env import OffloadingEnv
from ddqn_agent import DDQNAgent
import matplotlib.pyplot as plt
import time


def run(user=10, lamd=1.1, host="", port_base=10000, docker_min_max=[], duration=1000):
    system_arrival_rate = user * lamd
    system_inter_arrival_rate = 1 / system_arrival_rate
    cnt = 0
    agent = DDQNAgent(21, 3)

    # vì OffloadingEnv có async nên ta tạo 1 loop riêng trong thread
    env = asyncio.run(OffloadingEnv(num_servers=3).ainit()) if hasattr(OffloadingEnv, "ainit") else OffloadingEnv(num_servers=3)
    # env.ainit() 
    obs = asyncio.run(env.get_observation())
    queue = {}
    rewards = []
    all_reward = {}

    done = False
    
    
    while duration > 0:
        event = np.random.exponential(system_inter_arrival_rate)
        time_sleep = event if event > 0 else 0
        # ngủ sync luôn
        time.sleep(time_sleep)

        # chọn docker
        slected_docker = np.random.randint(docker_min_max[0], docker_min_max[1])
        slected_docker = agent.act(obs) + 1
        # if not (docker_min_max[0] <= slected_docker < docker_min_max[1]):
        #     raise ValueError("action is out of the selected range")

        slected_port = port_base + slected_docker
        
        cmd = [
            sys.executable,
            "host_send_request.py",
            "--request", "inference",
            "--num", "1",
            "--docker", str(slected_docker),
            "--port", str(slected_port),
            "--id", str(cnt)
        ]
        print(cmd)
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
                        print(train_data)
                        if re_val > 3:
                            done = True
                        agent.remember(train_data[0], train_data[1], train_data[2], train_data[3], done)
                                            
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

        duration -= event
        cnt += 1


if __name__ == "__main__":
    run(2, 1, "", 10000, [1, 4], 2000)



