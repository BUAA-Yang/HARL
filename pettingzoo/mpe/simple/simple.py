# noqa
"""
# Simple

```{figure} mpe_simple.gif
:width: 140px
:name: simple
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_v2` |
|--------------------|----------------------------------------|
| Actions            | Discrete/Continuous                    |
| Parallel API       | Yes                                    |
| Manual Control     | No                                     |
| Agents             | `agents= [agent_0]`                    |
| Agents             | 1                                      |
| Action Shape       | (5)                                    |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))        |
| Observation Shape  | (4)                                    |
| Observation Values | (-inf,inf)                             |
| State Shape        | (4,)                                   |
| State Values       | (-inf,inf)                             |

```{figure} ../../_static/img/aec/mpe_simple_aec.svg
:width: 200px
:name: simple
```

In this environment a single agent sees a landmark position and is rewarded based on how close it gets to the landmark (Euclidean distance). This is not a multiagent environment, and is primarily intended for debugging purposes.

Observation space: `[self_vel, landmark_rel_position]`

### Arguments

``` python
simple_v2.env(max_cycles=25, continuous_actions=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np

from pettingzoo.utils.conversions import parallel_wrapper_fn
from gymnasium.utils import EzPickle
from .._mpe_utils.core import Agent, Landmark, Obstacles,World
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(self,local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode=None):
        EzPickle.__init__(
            self,
            local_ratio,
            max_cycles,
            continuous_actions,
            render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world()
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 10
        world.dim_p = 3
        num_agents = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_agents)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size=0.05
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = False
            agent.size =0.1
            agent.goal=world.landmarks[i]
        # add obstacles
        world.obstacles = [Obstacles() for i in range(1)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = "obstacle %d" % i
            obstacle.collide = True
            obstacle.movable = False
            obstacle.color = np.array([0.5, 0.5, 0.5])  # Set obstacle color to gray
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # Set agent colors with a gradient
            agent.color = np.array([0.25 + 0.15 * i, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # Set landmark colors with a corresponding gradient
            landmark.color = np.array([0.25 + 0.15 * i, 0.25, 0.25])
        
        # set random initial states
        for i, agent in enumerate(world.agents):
            # Generate random spherical coordinates
            z = np_random.uniform(-0.3, 0.3)  # height restriction
            r = np.sqrt(1 - z**2)  # radius at this height
            phi = np_random.uniform(0, 2 * np.pi)  # azimuthal angle
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            agent.state.p_pos = np.array([x, y, z])
            agent.state.p_vel = np.zeros(world.dim_p)                                        
            agent.state.p_vel_horizontal = np.zeros(1)                                     #初始化水平速度为0
            agent.state.c = np.zeros(world.dim_c)
            agent.state.omega  = np_random.uniform(0, 2 * np.pi)                     #随机初始化航向角
        
        for i, landmark in enumerate(world.landmarks):
            # Place the landmark on the opposite side of the agent
            agent_pos = world.agents[i].state.p_pos
            landmark.state.p_pos = -agent_pos  # Opposite position
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, obstacle in enumerate(world.obstacles):
            # Place the obstacle at the center (0, 0, 0)
            obstacle.state.p_pos = np.array([0, 0, 0])
            # Place the obstacle randomly within a spherical shell of radius 10 to 20
            obstacle.size = np_random.uniform(0.2, 0.4)
    def is_collision(self, entity1, entity2, is_obstacle=False):
        if is_obstacle:
            # Calculate horizontal Euclidean distance (ignoring z-coordinate)
            delta_pos = entity1.state.p_pos[:2] - entity2.state.p_pos[:2]
        else:
            # Calculate full 3D Euclidean distance
            delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min else False
    def compute_relative_bearing_and_distance(self,agent_pos, target_pos, omega):
        """
        计算目标相对智能体的水平方位角、水平距离和垂直距离

        参数:
        - agent_pos: 智能体的位置 [x, y, z]
        - target_pos: 目标的位置 [x, y, z]
        - omega: 智能体的水平朝向角 (弧度)

        返回:
        - relative_bearing: 目标相对于智能体的水平方位角 [-pi, pi] (弧度)
        - horizontal_distance: xy 平面上的水平距离
        - vertical_distance: z 方向的垂直距离
        """
        # 计算相对位置向量
        delta_pos = np.array(target_pos) - np.array(agent_pos)
        # 水平距离（忽略 z）
        horizontal_distance = np.linalg.norm(delta_pos[:2])  # sqrt(dx^2 + dy^2)
        # 垂直距离（z轴方向）
        vertical_distance = delta_pos[2]
        # 计算绝对水平方向（xy平面上的方位角）[-pi, pi]
        absolute_bearing = np.arctan2(delta_pos[1], delta_pos[0])
        # 计算相对方位角（目标相对 agent 当前朝向 omega 的水平角度） ∈ [-pi, pi]
        relative_bearing = absolute_bearing - omega
        # 规范化到 [-pi, pi]，保持一致性
        relative_bearing = (relative_bearing + np.pi) % (2 * np.pi) - np.pi
        return relative_bearing, horizontal_distance, vertical_distance

    def reward(self, agent, world):
        rew = 0
        for a in world.agents:
            if a is not agent and self.is_collision(a, agent):
                rew -= 15
        for obstacle in world.obstacles:
            if self.is_collision(obstacle, agent, is_obstacle=True):
                rew -= 10
        # Reward based on the distance to the target landmark
        distance_to_goal = np.linalg.norm(agent.state.p_pos - agent.goal.state.p_pos)
        if distance_to_goal < agent.size:
            rew += 1
        return rew
    def global_reward(self, world):
        dist=0
        for agent in world.agents:
            # Calculate the squared distance to the agent's goal (target landmark)
            dist -= np.linalg.norm(agent.state.p_pos - agent.goal.state.p_pos)
        avg_dist = dist / len(world.agents)
        return avg_dist
    def observation(self, agent, world):
        # Get positions of the nearest agent and obstacle relative to this agent
        nearest_agent = None
        nearest_obstacle = None
        min_agent_dist = float('inf')
        min_obstacle_dist = float('inf')

        for other_agent in world.agents:
            if other_agent is not agent:
                dist = np.linalg.norm(other_agent.state.p_pos - agent.state.p_pos)
                if dist < min_agent_dist:
                    min_agent_dist = dist
                    nearest_agent = other_agent
        for obstacle in world.obstacles:
            # Calculate horizontal Euclidean distance (ignoring z-coordinate)
            dist = np.linalg.norm(obstacle.state.p_pos[:2] - agent.state.p_pos[:2])
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
                nearest_obstacle = obstacle
        # Relative positions of the nearest agent and obstacle
        nearest_agent_pos = (
            self.compute_relative_bearing_and_distance(agent.state.p_pos, nearest_agent.state.p_pos, agent.state.omega) if nearest_agent else np.zeros(world.dim_p)
        )
        nearest_obstacle_pos = (
            np.concatenate([self.compute_relative_bearing_and_distance(agent.state.p_pos, nearest_obstacle.state.p_pos, agent.state.omega)[:2], [nearest_obstacle.size]]) 
            if nearest_obstacle else np.zeros(world.dim_p)
        )
        # Relative position of the target landmark
        target_landmark_pos = self.compute_relative_bearing_and_distance(agent.state.p_pos, agent.goal.state.p_pos, agent.state.omega)
        # communication of nearest_agent
        comm = nearest_agent.state.c
        return np.concatenate([agent.state.p_vel,target_landmark_pos, nearest_agent_pos, nearest_obstacle_pos, comm])
