import numpy as np
from myenv import MyEnv
import math
from tqdm import tqdm

class CartPoleSolver():
    
    def __init__(self, gamma=0.98, epsilon=0.99, alpha=0.25, episodes=2000, batch_size=1000, interval_num=50):
        self.env = MyEnv()
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # 贪婪策略参数
        self.alpha = alpha # 学习率
        self.episodes = episodes # 决策序列长度
        self.batch_size = batch_size # 训练次数
        self.interval_num = interval_num # 连续变量转离散变量分为几段

        self.pa_bin = np.linspace(-math.pi, math.pi, interval_num+1)[1: -1]
        self.pv_bin = np.linspace(-math.pi*15, math.pi*15, interval_num+1)[1: -1]

        # self.q_table = np.random.uniform(low=0, high=1, size=(interval_num**2, 3))
        self.q_table = np.zeros((interval_num**2, 3), dtype= np.float64)
        
    def get_state_index(self, observation):
        pole_angle, pole_v = observation
        
        state_index = 0
        state_index += np.digitize(pole_angle, bins = self.pa_bin) * self.interval_num
        state_index += np.digitize(pole_v, bins = self.pv_bin)
        
        return state_index
    
    def update_Q_table(self, observation, action, reward, next_observation):        
        state_index = self.get_state_index(observation)
        next_state_index = self.get_state_index(next_observation)
        
        max_next = max(self.q_table[next_state_index][:])
        q_target = reward + self.gamma * max_next
        self.q_table[state_index, action] = self.q_table[state_index, action] + self.alpha * (q_target - self.q_table[state_index, action])
        
    def decide_action(self, observation, epsilon):
        
        state = self.get_state_index(observation)
        
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(3)
            
        return action
    def run(self, epsilon=0.1, quiet=True):
        observation = self.env.reset()
        # epsilon = self.epsilon * (1 / (episode + 1))
        for t in range(self.batch_size):
            if not quiet:
                self.env.render()
                # print(observation)
            # action = self.decide_action(observation, self.epsilon)
            action = self.decide_action(observation, epsilon)
            next_observation, reward, _, _ = self.env.step(action)
            self.update_Q_table(observation, action, reward, next_observation)
            observation = next_observation
    
    def solve(self):
        epsilon = self.epsilon
        for episode in tqdm(range(self.episodes)):
            epsilon *= self.epsilon
            self.run(epsilon)
            
    def get_Q_table(self):
        for i in range(3):
            print('action: ', i)
            for j in range(self.interval_num**2):
                a = int(j/self.interval_num)
                b = j%self.interval_num
                print('angel: ', a, ', angel_v: ', b)
                print(self.q_table[j, i])
    
a = CartPoleSolver()
a.solve()
a.run(quiet=False)
# a.get_Q_table()