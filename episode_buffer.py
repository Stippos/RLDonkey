from collections import deque

class EpisodeBuffer:

    def __init__(self, maxlen, discount):
        self.q = deque(maxlen=maxlen)
        self.d = discount

    def add(self, obs):
        if len(self.q) == self.q.maxlen:
            out = self.q.popleft()
            self.q.append(obs)

            reward = out[2][0]
            
            for i, s in enumerate(self.q):
                reward += self.d**i * s[2][0]

            norm = (1 - self.d**self.q.maxlen) / (1 - self.d) + 1
            out[2][0] = reward / norm
        
            return out

        else:
            self.q.append(obs)
            return None

    def as_list(self):
        return list(self.q)

        


