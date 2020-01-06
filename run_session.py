import numpy as np
import torch
import os
import time

from sac import SAC
from car import Car

from gym import spaces

from functions import process_image, image_to_ascii, rgb2gray
from episode_buffer import EpisodeBuffer
import db

def run_session(db_name, max_session_length, sweep, session, model_name, params):

    alg = SAC(params)
    car = Car()
    car.reset()

    training_after_episodes =  params["training_after_episodes"]

    episode = 0

    random_episodes = params["random_episodes"]

    max_episode_length = params["max_episode_length"]

    THROTTLE_MAX = params["throttle_max"]
    THROTTLE_MIN = params["throttle_min"]
    STEER_LIMIT_LEFT = -1
    STEER_LIMIT_RIGHT = 1

    action_space = spaces.Box(low=np.array([STEER_LIMIT_LEFT, -1]), 
        high=np.array([STEER_LIMIT_RIGHT, 1]), dtype=np.float32 )

    for i in range(max_session_length):
        episode += 1
        throttle = 0.15
        try:
            step = 0

            state = car.reset()
            time.sleep(1)
            state = car.step([0,0.01])
            #print(state)
            

            state = alg.process_image(state)
            state = np.stack((state, state, state, state), axis=0)
            episode_buffer = EpisodeBuffer(alg.horizon, alg.discount)
            episode_reward = 0

            while step < max_episode_length:
                
                t = time.time_ns()
                
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
                

                next_state = car.step(action)

                im = next_state

                darkness = len(im[(im > 120) * (im < 130)])

                if darkness < 2500:# < len(im[(im > 160) * (im < 170)]):
                    raise KeyboardInterrupt

                
                next_state = alg.process_image(next_state)
                reward = (throttle - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN) 

                reward = darkness / 7000

                image_to_ascii(next_state[::2].T)

                episode_reward += reward
                print("Sweep: {}, Episode: {}, Step: {}, Episode reward: {:.2f}, Step reward: {:.2f}".format(sweep, episode, step, episode_reward, reward))

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

            db.insert_episode(db_name, session, episode, step, episode_reward)

            time.sleep(5)

