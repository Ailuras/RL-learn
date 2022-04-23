import numpy as np
from myenv import MyEnv
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class CartPoleSolver():
    
    def __init__(self, gamma=0.98, epsilon=0.5, alpha=0.01, episodes=2000, batch_size=1000, interval_num=20, error=1e-1):
        self.env = MyEnv()
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # 贪婪策略参数
        self.alpha = alpha # 学习率
        self.episodes = episodes # 决策序列长度
        self.batch_size = batch_size # 训练次数
        self.interval_num = interval_num # 连续变量转离散变量分为几段
        self.error = error
        self.error_table = []

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
            next_observation, reward, _, _ = self.env.step(action, self.interval_num)
            self.update_Q_table(observation, action, reward, next_observation)
            observation = next_observation
    
    def compute_error(self, a, b):
        # return np.linalg.norm(np.mat(a)-np.mat(b))
        error = a-b
        return (error*error).max()
    
    def solve(self):
        epsilon = self.epsilon
        self.index = 0
        while True:
            self.index += 1
            q_table = self.q_table.copy()
            epsilon = self.epsilon/self.index
            self.run(epsilon)
            error = self.compute_error(q_table, self.q_table)
            self.error_table.append(error)
            if error < self.error:
                print('经过%d次迭代后收敛'%self.index)
                break
            else:
                print('第%d次迭代，误差为%f'%(self.index, error))
                print(epsilon)
                continue
            
    def get_Q_table(self):
        for i in range(3):
            print('action: ', i)
            for j in range(self.interval_num**2):
                a = int(j/self.interval_num)
                b = j%self.interval_num
                print('angel: ', a, ', angel_v: ', b)
                print(self.q_table[j, i])

    def plot_error(self):
        x = range(self.index)
        y = self.error_table
        plt.title("Error vs. Iteration Plot")
        plt.plot(x, y, label="Train_Loss_list")
        plt.xlabel("iteration")
        plt.ylabel("error")
        plt.show()
    
    def plot_Q_table(self):
        fig = plt.figure(figsize=(18, 6), facecolor='w')
        x = np.arange(self.interval_num)
        y = np.arange(self.interval_num)
        X, Y = np.meshgrid(x, y)
        Z1 = self.q_table[:, 0].reshape(self.interval_num, -1)
        Z2 = self.q_table[:, 1].reshape(self.interval_num, -1)
        Z3 = self.q_table[:, 2].reshape(self.interval_num, -1)
        
        ax = fig.add_subplot(131, projection='3d')
        plt.title("Q table of action -15")
        ax.plot_surface(X, Y, Z1, cmap='rainbow')
        ax.contour(X, Y, Z1, zdim='z', offset=0, cmap='rainbow')
        
        ax = fig.add_subplot(132, projection='3d')
        plt.title("Q table of action 0")
        ax.plot_surface(X, Y, Z2, cmap='rainbow')
        ax.contour(X, Y, Z2, zdim='z', offset=0, cmap='rainbow')
        
        ax = fig.add_subplot(133, projection='3d')
        plt.title("Q table of action 15")
        ax.plot_surface(X, Y, Z3, cmap='rainbow')
        ax.contour(X, Y, Z3, zdim='z', offset=0, cmap='rainbow')
        
        plt.show()
            
    def plot_action_of_state(self):
        # plt.title("action of state")
        q1 = self.q_table[:, 0]
        q2 = self.q_table[:, 1]
        q3 = self.q_table[:, 2]
        action = np.maximum(q1, q2)
        action = np.maximum(action, q3).reshape(self.interval_num, -1)
        plt.matshow(action, cmap='rainbow')
        plt.show()

    
a = CartPoleSolver()
a.solve()
a.run(quiet=False)
a.plot_error()
a.plot_action_of_state()
a.plot_Q_table()