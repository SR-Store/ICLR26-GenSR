import time

class Timer:
    def __init__(self):
        self.count = 0
        self.time = 0
        self.start_time = time.time()

    def add(self, n):
        self.count += n
        self.time += time.time() - self.start_time
        self.start_time = time.time()

    def pop(self, reset=True):
        speed = self.count / self.time
        if reset: self.count = self.time = 0
        return speed


class CycleTimer(Timer):
    def __init__(self, N):
        self.N = N
        self.time = [0] * N
        self.start_time = time.time()

    def add(self, idx):
        self.time[idx] += time.time() - self.start_time
        self.start_time = time.time()

    def pop(self, reset=True):
        total = sum(self.time)
        ratio = [t / total for t in self.time]
        if reset: self.time = [0] * self.N
        return total, ratio


class NamedTimer(Timer):
    def __init__(self):
        self.time = {}
        self.count = {}
        self.start_time = time.time()

    def __str__(self):
        total_time = sum(self.time.values())
        return f'{total_time*1000:.0f}ms' + ' (\n' + '\n'.join(
                   f'  {k:30} : {self.time[k] / total_time:3.0%} = {self.count[k]:5} * {self.time[k] / self.count[k] * 1000 ** 2:5,.0f}us'
                   for k in sorted(self.time, key=self.time.get, reverse=True)
               ) + '\n)'

    def add(self, name):
        if name not in self.time: self.time[name] = self.count[name] = 0
        self.time[name] += time.time() - self.start_time
        self.count[name] += 1
        self.start_time = time.time()

    def pop(self, reset=True):
        res = self.time
        if reset: self.time = {}
        return res

    def total(self):
        return sum(self.time.values())

