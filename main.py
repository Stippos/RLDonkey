import numpy as np
import torch
import os
import time

from sac import SAC
from car import Car

from gym import spaces

from functions import process_image, image_to_ascii, rgb2gray

from episode_buffer import EpisodeBuffer

alg = SAC()
car = Car()
car.reset()

## SAC hyperparameters
gamma = 0.99
tau = 0.005
lr = 0.001
hidden_size = 256
batch_size = 64
n_random_episodes = 10
discount = 0.9
horizon = 50
im_rows = 40
im_cols = 40
linear_output = 64


## Other hyperparameters

training_after_episodes = 1



episode = 0
random_episodes = 5

cmd = input("If you want to load a model, give model path, default last checkpoint.")
if cmd != "":
    episode = random_episodes
    if os.path.isfile(cmd):
        alg = torch.load(cmd)
    else:
        alg = torch.load("sac_model_checkpoint.pth")

max_episode_length = 5000
THROTTLE_MAX = 0.3
THROTTLE_MIN = 0.1
STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1


action_space = spaces.Box(low=np.array([STEER_LIMIT_LEFT, -1]), 
high=np.array([STEER_LIMIT_RIGHT, 1]), dtype=np.float32 )


for i in range(1000):
    #input("Press enter to start")      
    episode += 1
    throttle = 0.15
    try:
        step = 0
#       state, info = car.reset()

        state = car.reset()
        car.step([0,0.01])
        time.sleep(1)
        state = alg.process_image(state)
        state = np.stack((state, state, state, state), axis=0)
        episode_buffer = EpisodeBuffer(alg.horizon, alg.discount)
        episode_reward = 0

        while step < max_episode_length:
            t = time.time_ns()
            #print(state)
            step += 1
            temp = state[np.newaxis, :]

            if episode < random_episodes:
                action = action_space.sample()
            else:
                action = alg.select_action(temp)
                #action[1] = max(THROTTLE_MIN, min(THROTTLE_MAX, action[1]))
                action[0] = max(STEER_LIMIT_LEFT, min(STEER_LIMIT_RIGHT, action[0]))
            
            throttle += action[1] / 100.0
            throttle = max(THROTTLE_MIN, min(THROTTLE_MAX, throttle))
            action[1] = throttle
            action[1] = 0.3
            

#           next_state, info = car.step(action)
            next_state = car.step(action)

            im = next_state

            darkness = len(im[(im > 120) * (im < 130)])
            if darkness < 2500:# < len(im[(im > 160) * (im < 170)]):
                raise KeyboardInterrupt

            # if info["cte"] > 2.5 or info["cte"] < -2:
            #     raise KeyboardInterrupt

            next_state = alg.process_image(next_state)
            #reward = float(len(next_state[np.isclose(next_state, state[3, :, :], atol=1.5)]) / 1600.0)
            reward = (throttle - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN) 

            reward = darkness / 7000

            image_to_ascii(next_state[::2].T)

            episode_reward += reward
            print("Episode: {}, Step: {}, Episode reward: {:.2f}, Step reward: {:.2f}".format(episode, step, episode_reward, reward))

            not_done = 1.0

            next_state = next_state[np.newaxis, :]
            next_state = np.vstack((state[:3, :, :], next_state))

            out = episode_buffer.add([state, action, [reward], next_state, [not_done]])

            last = [state, action, [reward], next_state, [not_done]]
            alg.push_buffer(last)
            
            #if out:
                #alg.push_buffer(out)

            state = next_state

            if len(alg.replay_buffer) > alg.batch_size:
               alg.update_parameters()

            tn = time.time_ns()

            #sync with the network
            time.sleep(max(0, 0.1 - (tn - t) / 1e9))

        raise KeyboardInterrupt

    except KeyboardInterrupt:
        
        last[4] = [0]
        alg.push_buffer(last)

        car.reset()
        
        #if episode % 5 == 0:
            #print("Saving chekcpoint")
            #torch.save(alg, "sac_model_checkpoint.pth")
        print("Calculating reward")

        # episode_buffer = episode_buffer.as_list()

        # for i in range(len(episode_buffer)):
        #     reward = 0
        
        #     for j in range(min(len(episode_buffer) - i, alg.horizon)):
        #         reward += alg.discount**j * episode_buffer[i + j][2][0]
            
        #     norm = (1 - alg.discount**alg.horizon) / (1 - alg.discount)
        #     e = episode_buffer[i]
        #     e[2] = [reward / norm]
        #     if i == len(episode_buffer) - 1:
        #         e[-1][0] = 0.0

        #     alg.push_buffer(e)

        if len(alg.replay_buffer) > alg.batch_size:
            print("Training")
            for i in range(training_after_episodes):
                alg.update_parameters()

        time.sleep(5)
        

