import time
import numpy as np
import cv2
import torch
import random

from car import Car
from sac_with_vae import SAC
from vae import VAE

from gym import spaces



pretrained_vae = False
#vae = torch.load("vae.pth")
vae = VAE(lr=0.0001)

vae_training_episodes = 0
vae_initial_updates = 100
vae_updates = 1
vae_train_size = 1000

#vae_output = vae.linear_output
vae_output = 32
frame_stack = 1 
len_command_history = 9
sac_input = (vae_output + 2 + len_command_history * 2) * frame_stack

memory_discount = 0.95
memory_horizon = 1

sac_params = {
        "linear_output": sac_input,
        "lr": 0.0003,
        "target_entropy": -2,
        "batch_size": 64,
        "hidden_size": 64
        
        }

# Create the controller for the Donkey env
env = Car("kari_main", "mqtt.eclipse.org")
env.reset()
# Create the SAC agent to control the env
agent = SAC(parameters = sac_params)
# Create the state representation functionality


throttle_weight_1 = 0.1
throttle_weight_2 = -5

STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1
THROTTLE_MAX = 0.45
THROTTLE_MIN = 0.35
MAX_STEERING_DIFF = 0.3

action_space = spaces.Box(
        low=np.array([STEER_LIMIT_LEFT, THROTTLE_MIN]), 
        high=np.array([STEER_LIMIT_RIGHT, THROTTLE_MAX]), dtype=np.float32)

episodes = 10000
max_steps = 1000
n_random_episodes = 10

agent_training_episodes = 600 
vae_training_episodes = 600


def is_dead(im, threshold):
    darkness = len(im[(im > 120) * (im < 130)])

    if darkness < threshold:
        return True
    else:
        return False
def darkness(im):
    return len(im[(im > 120) * (im < 130)])

def enforce_limits(action, prev_steering):
     var = (THROTTLE_MAX - THROTTLE_MIN) / 2
     mu = (THROTTLE_MAX + THROTTLE_MIN) / 2
     
     steering_min = max(STEER_LIMIT_LEFT, prev_steering - MAX_STEERING_DIFF)
     steering_max = min(STEER_LIMIT_RIGHT, prev_steering + MAX_STEERING_DIFF)
     
     steering = max(steering_min, min(steering_max, action[0]))
     #print("Prev steering: {:.2f}, Steering min: {:.2f}, Steering max: {:.2f}, Action: {:.2f}, Steering: {:.2f}".format(prev_steering, steering_min, steering_max, action[0], steering))
     return (steering, action[1] * var + mu)

def image_to_ascii(im, size):

    gs_im = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    im = cv2.resize(gs_im, (size*2, size)).T
    asc = []
    chars = ["B","S","#","&","@","$","%","*","!",":","."]
    for j in range(im.shape[1]):
        line = []
        for i in range(im.shape[0]):
            line.append(chars[int(im[i, j]) // 25])
        asc.append("".join(line))

    for line in asc:
        print(line)


for e in range(episodes):
    
    episode_reward = 0
    step = 0
    done = 0.0 
    episode_buffer = []
    command_history = np.zeros(2*len_command_history)
    # set-up the env for a new episode
    steering = 0
    throttle = 0.01
    action = [steering, throttle]

    env.reset()
    env.step(action)
    time.sleep(3)

    obs = env.step([0, 0.01])
    
    #form the state space
    #embedding should be a one dimensional (n, ) numpy array 
    embedding = vae.embed(obs)
    #print(embedding.shape)
    #print(embedding.shape)
    #print(action.shape())
    state_action = np.hstack((embedding, action, command_history))
    image_action = np.hstack((obs.reshape(-1), action, command_history))
    
    state = np.hstack([state_action for x in range(frame_stack)])
    image_state = np.hstack([image_action for x in range(frame_stack)])
    #print(image_state.shape)
    while step < max_steps:
        step += 1
        t1 = time.time_ns()
        
        if e < n_random_episodes:
            action = action_space.sample()
        else:
            action = agent.select_action(state[np.newaxis, :])

        taken_action = enforce_limits(action, command_history[0])
        obs = env.step(taken_action)
        vae.add_image(obs)
        command_history = np.roll(command_history, 2)
        command_history[:2] = taken_action
        #print(action)
        #print(command_history)
        reward = 1 + throttle_weight_1 * (action[1] + 1)
        reward = darkness(obs) / 7000
        

        if is_dead(obs, 2500):
            done = 1.0
            reward = -10 + throttle_weight_2 * (action[1] + 1)

        state_action = np.hstack((embedding, action, command_history))
        image_action = np.hstack((obs.reshape(-1), action, command_history))
    
        #print(state_action.shape)
        #print(state[:(vae_output + 2) * (frame_stack - 1)].shape)
        next_state = np.hstack([state_action, state[:(vae_output + 2) * (frame_stack - 1)]])
        next_image_state = np.hstack([image_action, state[:(vae_output + 2) * (frame_stack - 1)]])
        #print(next_state.shape)
        
        episode_buffer.append([image_state, action, [reward], next_image_state, [float(not done)]])
        #agent.push_buffer([state, action, [reward], next_state, [float(not done)]])
       
        episode_reward += reward
        t2 = time.time_ns()
        image_to_ascii(obs, 20)
        print("Episode: {}, Step: {}, Reward: {:.2f}, Episode reward: {:.2f}, Time: {:.2f}".format(e, step, reward, episode_reward, (t2 - t1) / 1e6))
        t1 = t2
        state = next_state
        image_state = next_image_state
        if done:
            break
    
    env.reset()
    time.sleep(3)
    env.step((0,0.01))

    for i in range(len(episode_buffer)):
        for j in range(min(memory_horizon, len(episode_buffer) - i)):
            reward += episode_buffer[i + j][2][0] * memory_discount
        ep = episode_buffer[i]
        ep[2][0] = reward
        agent.push_buffer(ep)


    if e >= n_random_episodes - 1:
        if not pretrained_vae:
            if e == n_random_episodes - 1:
                vae.update_parameters(vae_initial_updates, len(vae.images))
            elif e >= n_random_episodes:
                vae.update_parameters(vae_updates, vae_train_size)
    
        for i in range(agent_training_episodes):

            k = min(agent.batch_size, len(agent.replay_buffer))
            batch = random.sample(agent.replay_buffer, k=k)

            for i in batch:
                im = i[0][:160*120*3]
                next_im = i[3][:160*120*3]

                if len(im) == 52:
                    continue
                embedding = vae.embed(im.reshape(160, 120,3)).squeeze()
                next_embedding = vae.embed(next_im.reshape(160, 120,3)).squeeze()
                
                state = np.hstack((embedding, i[0][160*120*3:]))
                next_state = np.hstack((next_embedding, i[0][160*120*3:]))
                i[0] = state
                i[3] = next_state

            agent.update_parameters(batch)


   
