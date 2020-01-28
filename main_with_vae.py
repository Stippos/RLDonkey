SAC, EncoderDeepConv, DecoderDeepConv
from car import Car

from gym import spaces

from functions import process_image, image_to_ascii, rgb2gray

from episode_buffer import EpisodeBuffer

params = {
    "target_entropy": -2,
    "hidden_size": 64,
    "batch_size": 64,
    "gamma": 0.9,
    "discount": 0.95,
    "lr": 0.0003,
    "coder_lr": 0.001,
    "linear_output": 10,
    "im_rows": 80,
    "im_cols": 160,
    "horizon": 1
}

alg = SAC(parameters=params)
car = Car(car="kari_main")

alg.load_coder("decoder.pth", "encoder.pth")

car.reset()
time.sleep(2)

## SAC hyperparameters

## Other hyperparameters

training_after_episodes = 10

episode = 0
random_episodes = 2


max_episode_length = 5000
THROTTLE_MAX = 0.5
THROTTLE_MIN = 0.25
STEER_LIMIT_LEFT = -0.5
STEER_LIMIT_RIGHT = 0.5

c_loss = a_loss = None

action_space = spaces.Box(low=np.array([STEER_LIMIT_LEFT, THROTTLE_MIN]), 
high=np.array([STEER_LIMIT_RIGHT, THROTTLE_MAX]), dtype=np.float32 )


for i in range(1000):
    #input("Press enter to start")      
    episode += 1
    throttle = 0.1

    #if episode == random_episodes + 10:
    #    alg.update_sac_lr(params["lr"] / 10)
    
    try:
        step = 0
#       state, info = car.reset()

        state = car.reset()

        im = state
        darkness = len(im[(im > 120) * (im < 130)])
        steering = 0
        throttle = 0.01
        car.step([steering,throttle])
        time.sleep(1)
        state = alg.process_image(state)[np.newaxis,: ]
        #print(state.shape)
        #state = np.stack((state, state, state, state), axis=0)
        episode_buffer = EpisodeBuffer(alg.horizon, alg.discount)
        episode_buffer = []
        episode_reward = 0

        while step < max_episode_length:
            t = time.time_ns()
            #print(state)
            step += 1
            temp = state[np.newaxis, :]
            #print(temp.shape)
            if episode < random_episodes:
                action = action_space.sample()
            else:
                steering = alg.select_action(temp, throttle, steering)
                #action[1] = max(THROTTLE_MIN, min(THROTTLE_MAX, action[1]))
                action[0] = max(STEER_LIMIT_LEFT, min(STEER_LIMIT_RIGHT, action[0]))
            
            #throttle += action[1] / 100.0
            new_throttle = max(THROTTLE_MIN, min(THROTTLE_MAX, action[1]))
            action[1] = throttle
            new_steering = action[0]
            #action[1] = 0.3
            

#           next_state, info = car.step(action)
            next_state = car.step(action)

            im = next_state

            new_darkness = len(im[(im > 120) * (im < 130)])
            if new_darkness < 2500:# < len(im[(im > 160) * (im < 170)]):
                raise KeyboardInterrupt

            # if info["cte"] > 2.5 or info["cte"] < -2:
            #     raise KeyboardInterrupt

            next_state = alg.process_image(next_state)
            #reward = float(len(next_state[np.isclose(next_state, state[3, :, :], atol=1.5)]) / 1600.0)
            reward = (throttle - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN)

            #reward = (new_darkness - darkness) / 500
            darkness = new_darkness

            image_to_ascii(next_state[::4,::4].T)

            episode_reward += reward
            print("Episode: {}, Step: {}, Episode reward: {:.2f}, Step reward: {:.2f}".format(episode, step, episode_reward, reward))
            print("Critic loss: {}, Actor loss: {}".format(c_loss, a_loss))
            not_done = 1.0

            next_state = next_state[np.newaxis, :]
            #next_state = np.vstack((state[:3, :, :], next_state))

            #out = episode_buffer.add([state, action, [reward], next_state, [not_done]])

            #last = [state, action, [reward], next_state, [not_done]]
            
            last = [state, action, [reward], next_state, [not_done]]
            episode_buffer.append(last)
            #alg.push_buffer(last)
            
            #if out:
                #alg.push_buffer(out)

            state = next_state

            #if len(alg.replay_buffer) > alg.batch_size and episode >= random_episodes:
               #c_loss, a_loss = alg.update_parameters()

            tn = time.time_ns()

            #sync with the network
            time.sleep(max(0, 0.1 - (tn - t) / 1e9))

        raise KeyboardInterrupt

    except KeyboardInterrupt:

        #last[4] = [0]
        #alg.push_buffer(last)

        car.reset()
        
        #if episode % 5 == 0:
            #print("Saving chekcpoint")self.im_cols
        #episode_buffer = episode_buffer.as_list()

        for i in range(len(episode_buffer)):
            reward = 0
        
            for j in range(min(len(episode_buffer) - i, alg.horizon)):
                reward += alg.discount**j * episode_buffer[i + j][2][0]
            
            norm = (1 - alg.discount**alg.horizon) / (1 - alg.discount)
            e = episode_buffer[i]
            e[2] = [reward / norm]
            #e[2] = [step]
            if i == len(episode_buffer) - 1:
                e[-1][0] = 0.0

            alg.push_buffer(e)


        #if episode == random_episodes - 1:

            #print("Training vae")
            #alg.update_encoder(epochs=100, size=len(alg.replay_buffer))
            #alg.update_coder_lr(0.0001)

        #if episode >= random_episodes:
            #print("Training vae")
            #alg.update_encoder(epochs=10, size=min(1000, len(alg.replay_buffer)))
        
        if len(alg.replay_buffer) > alg.batch_size and episode >= random_episodes:
            print("Training")
            c_loss = a_loss = 0
            for i in range(training_after_episodes):
                c, a = alg.update_parameters()
                c_loss += c
                a_loss += a

            c_loss /= training_after_episodes
            a_loss /= training_after_episodes

        time.sleep(4)
        

