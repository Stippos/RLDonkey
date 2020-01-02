from donkeycar.gym import remote_controller
import numpy as np
import time

class Car:
    
    def __init__(self, car = "kari", server = "mqtt.eclipse.org"):
        self.control = remote_controller.DonkeyRemoteContoller(car, server)
        self.state = self.control.observe()
        #self.size = np.prod(self.state.shape)

    def reset(self):
        self.control.take_action(action=[0, 0])
        #self.state, info = self.control.observe()
        self.state = self.control.observe()
        
        return self.state#, info
    
    def step(self, control):
        # new_state = self.control.observe()
        # self.size = np.prod(new_state.shape)
        # if len(self.state[np.isclose(self.state, new_state)]) == self.size:
        #     time.sleep(0.001)
        #     self.step(control)
        
        self.control.take_action(action=control)
        return self.control.observe()