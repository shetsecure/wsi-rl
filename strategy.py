import math


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(
            -1.0 * current_step * self.decay
        )


class DeterministicStrategy:
    def __init__(self, rate=0):
        self.rate = rate
        
    def get_exploration_rate(self, current_step):
        return self.rate