import gym
import pygame
import sys
env = gym.make('CartPole-v1')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

# pygame.quit()
# sys.exit()
