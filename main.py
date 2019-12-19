import numpy as np

from sac import SAC
from car import Car

from gym import spaces

from functions import process_image, image_to_ascii, rgb2gray

alg = SAC()
car = Car()

max_episode_length = 5000
THROTTLE_MAX = 0.25
THROTTLE_MIN = 0.15
STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1
episode = 0

action_space = spaces.Box(low=np.array([STEER_LIMIT_LEFT, -1]), 
high=np.array([STEER_LIMIT_RIGHT, 1]), dtype=np.float32 )

episode_buffer = []

for i in range(1000):
    input("Press enter to start:")
    episode += 1
    throttle = 0.15
    try:
        step = 0
        state = car.reset()
        state = alg.process_image(state)
        state = np.stack((state, state, state, state), axis=0)
        episode_buffer = []
        episode_reward = 0

        while step < max_episode_length:
            print(state)
            step += 1
            temp = state[np.newaxis, :]

            if episode < 3:
                action = action_space.sample()
            else:
                action = alg.select_action(temp)
                #action[1] = max(THROTTLE_MIN, min(THROTTLE_MAX, action[1]))
                action[0] = max(STEER_LIMIT_LEFT, min(STEER_LIMIT_RIGHT, action[0]))
            
            throttle += action[1] / 100.0
            throttle = max(THROTTLE_MIN, min(THROTTLE_MAX, throttle))
            action[1] = throttle

            next_state = alg.process_image(car.step(action))
            #reward = float(len(next_state[np.isclose(next_state, state[3, :, :], atol=1.5)]) / 1600.0)
            reward = (throttle - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN) 

            image_to_ascii(next_state[::2].T)

            episode_reward += reward
            print("Episode: {}, Step: {}, Episode reward: {:.2f}, Step reward: {:.2f}".format(episode, step, episode_reward, reward))

            not_done = 1.0

            next_state = next_state[np.newaxis, :]
            next_state = np.vstack((state[:3, :, :], next_state))

            episode_buffer.append([state, action, [reward], next_state, [not_done]])

            state = next_state

            if len(alg.replay_buffer) > alg.batch_size:
               alg.update_parameters()

        raise KeyboardInterrupt

    except:
        
        car.reset()
        for i in range(len(episode_buffer)):
            reward = 0
        
            for j in range(min(len(episode_buffer) - i, alg.horizon)):
                reward += alg.discount**j * episode_buffer[i + j][2][0]
            
            norm = (1 - alg.discount**alg.horizon) / (1 - alg.discount)
            e = episode_buffer[i]
            e[2] = [reward / norm]
            if i == len(episode_buffer) - 1:
                e[-1][0] = 0.0

            alg.push_buffer(e)

        for i in range(50):
            print("Training: ")
            alg.update_parameters()

        

