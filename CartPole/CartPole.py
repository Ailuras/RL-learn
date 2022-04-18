import gym
env = gym.make('CartPole-v2')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done: # 如果结束, 则退出循环
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()