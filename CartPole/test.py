# from myenv import MyEnv

# env = MyEnv()

# for i_episode in range(1):
#     observation = env.reset()
#     for t in range(500):
#         env.render()
#         print(observation)
#         action = 0
#         observation, reward, done, info = env.step(action)
#         if done: # 如果结束, 则退出循环
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
import numpy as np

a = [[1,2], [1,3]]
b = [[2,3], [2,4]]
print(np.linalg.norm(np.mat(a)-np.mat(b)))