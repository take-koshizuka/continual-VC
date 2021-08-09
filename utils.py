import numpy as np

class EarlyStopping(object):
    def __init__(self, monitor='loss', direction='min'):
        self.monitor = monitor
        self.direction = direction
        self.best_state = None
        if direction == 'min':
            self.monitor_values = { self.monitor : float('inf') }
        elif direction == 'max':
            self.monitor_values = { self.monitor : -float('inf') }
        else:
            raise ValueError("args: [direction] must be min or max")

    def judge(self, values):
        return (self.direction == 'min' and self.monitor_values[self.monitor] > values[self.monitor]) \
                    or (self.direction == 'max' and self.monitor_values[self.monitor] < values[self.monitor])

    def update(self, values):
        self.monitor_values[self.monitor] = values[self.monitor]
