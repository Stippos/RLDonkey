from donkeycar.gym import remote_controller

class Car:
    
    def __init__(self, car = "kari", server = "mqtt.eclipse.org"):
        self.control = remote_controller.DonkeyRemoteContoller(car, server)

    def reset(self):
        self.control.take_action(action=[0, 0])
        return self.control.observe()
    
    def step(self, control):
        self.control.take_action(action=control)
        return self.control.observe()