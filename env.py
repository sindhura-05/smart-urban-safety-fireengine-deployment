import numpy as np

class FireEnv:
    def __init__(self, graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.reset()

    def reset(self):
        self.selected = set()
        return self._state()

    def _state(self):
        state = np.zeros(self.n)
        for v in self.selected:
            state[v] = 1
        return state

    def step(self, action):
        if action in self.selected:
            return self._state(), -1, False

        self.selected.add(action)
        covered = set(self.selected)
        for v in self.selected:
            covered |= set(self.graph.neighbors(v))

        reward = len(covered)
        done = len(self.selected) >= 3
        return self._state(), reward, done
