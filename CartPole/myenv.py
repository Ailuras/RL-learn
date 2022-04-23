from ast import Mod
import math
from typing import Optional, Union
import numpy as np
import pygame
from pygame import gfxdraw
import gym
from gym import spaces, logger
from gym.utils import seeding

class MyEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    新的环境状态仅包含角度[-pi, pi]和角速度[-15pi, 15pi]
    动作包含-3v, 0, 3v
    """
    def __init__(self):
        self.g = 9.81
        self.m = 0.055
        self.l = 0.042
        self.J = 1.91e-4
        self.b = 3e-6
        self.K = 0.0536
        self.R = 9.5
        self.T_s = 0.005
        self.umap = {0:-3, 1:0, 2:3}
        self.R_rew = 1
        self.Q_rew1 = 5
        self.Q_rew2 = 0.1

        self.a_threshold = 15 * math.pi
        high = np.array(
            [
                math.pi,
                self.a_threshold * 2,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def step(self, action, method):
        theta, theta_v = self.state
        u = self.umap[action]
        
        temp = 1/self.J * (self.m*self.g*self.l*math.sin(theta) - self.b*theta_v - self.K**2/self.R*theta_v + self.K/self.R*u)
        theta_new = theta + self.T_s*theta_v
        theta_v_new = theta_v + self.T_s*temp
        if theta_new > math.pi:
            theta_new -= math.pi*2
        elif theta_new < -math.pi:
            theta_new += math.pi*2
        self.state = (theta_new, theta_v_new)
        done = False
        if method == 0:
            reward = -self.Q_rew1*theta_new**2 - self.Q_rew2*theta_v_new**2 - self.R_rew*u**2
        else:
            theta_unit = math.pi*2/method
            theta_v_unit = math.pi*30/method
            theta_r = (int((theta_new+math.pi)/theta_unit) + 1/2) * theta_unit - math.pi
            theta_v_r = (int((theta_v_new+math.pi*15)/theta_v_unit) + 1/2) * theta_v_unit - math.pi*15
            reward = -self.Q_rew1*theta_r**2 - self.Q_rew2*theta_v_r**2 - self.R_rew*u**2

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = (math.pi, 0)
        # self.state = (0, 0)
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self):
        screen_width = 600
        screen_height = 400

        world_width = 4.8
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = screen_width / 2.0
        carty = 150
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[0])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        pygame.event.pump()
        self.clock.tick(50)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
