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

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv):
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
        num_agents = 4
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = False
            agent.size =2
        # add landmarks
        world.landmarks = [Landmark() for i in range(len(world.agents))]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # add obstacles
        world.obstacles = [Landmark() for i in range(1)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = "obstacle %d" % i
            obstacle.collide = True
            obstacle.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # Set agent colors with a gradient
            agent.color = np.array([0.25 + 0.15 * i, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # Set landmark colors with a corresponding gradient
            landmark.color = np.array([0.75, 0.25 + 0.15 * i, 0.25])
        
        # set random initial states
        for i, agent in enumerate(world.agents):
            # Generate random spherical coordinates
            z = np_random.uniform(-60, 60)  # height restriction
            r = np.sqrt(100**2 - z**2)  # radius at this height
            phi = np_random.uniform(0, 2 * np.pi)  # azimuthal angle
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            agent.state.p_pos = np.array([x, y, z])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        for i, landmark in enumerate(world.landmarks):
            # Place the landmark on the opposite side of the agent
            agent_pos = world.agents[i].state.p_pos
            landmark.state.p_pos = -agent_pos  # Opposite position
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, obstacle in enumerate(world.obstacles):
            # Place the obstacle at the center (0, 0, 0)
            obstacle.state.p_pos = np.array([0, 0, 0])
            # Place the obstacle randomly within a spherical shell of radius 10 to 20
            obstacle.size = np_random.uniform(10, 20)
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

    def reward(self, agent, world):
        rew = 0
        for a in world.agents:
            if a is not agent and self.is_collision(a, agent):
                rew -= 10
        for obstacle in world.obstacles:
            if self.is_collision(obstacle, agent, is_obstacle=True):
                rew -= 5
        
        return rew
    def global_reward(self, world):
        dist=0
        for agent in world.agents:
            # Find the corresponding landmark for the agent
            agent_index = int(agent.name.split('_')[1])
            target_landmark = world.landmarks[agent_index]
            # Calculate the squared distance to the corresponding landmark
            dist-= np.sum(np.square(agent.state.p_pos - target_landmark.state.p_pos))
        return dist
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
            nearest_agent.state.p_pos - agent.state.p_pos if nearest_agent else np.zeros(world.dim_p)
        )
        nearest_obstacle_pos = (
            np.concatenate([nearest_obstacle.state.p_pos[:2] - agent.state.p_pos[:2], [nearest_obstacle.size]]) 
            if nearest_obstacle else np.zeros(world.dim_p)
        )
        # Relative position of the target landmark
        agent_index = int(agent.name.split('_')[1])
        target_landmark = world.landmarks[agent_index]
        target_landmark_pos = target_landmark.state.p_pos - agent.state.p_pos
        return np.concatenate([agent.state.p_vel,target_landmark_pos, nearest_agent_pos, nearest_obstacle_pos])
