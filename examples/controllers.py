import numpy as np
import random
import pickle

class AgentController:
    def __init__(self, eps=0.5, gamma=0.99, lr=0.1, eps_decay=0.995, min_eps=0.01):
        self.q_table = {}
        self.eps = eps
        self.gamma = gamma
        self.lr = lr
        self.eps_decay = eps_decay
        self.min_eps = min_eps

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)

        return self.q_table[state]

    def update_q(self, state, next_state, action, reward, done=False):
        q_s = self.get_q(state)
        q_s_next = self.get_q(next_state)

        q_s_max = np.max(q_s_next) if not done else 0
        self.q_table[state][action] = q_s[action] + self.lr * (reward + self.gamma * q_s_max - q_s[action])

    def choose_action(self, state):
        if self.eps > random.random():
            return random.randint(0, 3)

        return np.argmax(self.get_q(state))

    def set_eps(self, eps):
        self.eps = eps

    def save_q(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

class MultiAgentController:
    def __init__(self, agent_ids, **kwargs):
        self.controllers = {a_id: AgentController(**kwargs) for a_id in agent_ids}

    def save_qs(self, save_path: str):
        q_tables = {a_id: con.q_table for a_id, con in self.controllers.items()}
        with open(save_path, 'wb') as f:
            pickle.dump(q_tables, f)

    def load_qs(self, path):
        with open(path, 'rb') as f:
            q_tables = pickle.load(f)
        for a_id, con in self.controllers.items():
            con.q_table = q_tables[a_id]