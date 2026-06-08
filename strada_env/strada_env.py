import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import random
import sys
import json
import multiprocessing as mp
from functools import partial
import logging

logger = logging.getLogger(__name__)

""" ------------------ REGULAR ENVIRONMENT ------------------ """

""" PREPARE MAP """
def prepare_map(map_file_path: str):
    city_map = []
    with open(map_file_path, 'r', encoding='utf-8') as f:
        rows = f.readlines()

    for row in rows:
        city_map.append([int(x) for x in row.split(' ')])

    # x, y (0-29)
    city_map = np.array(city_map).astype(np.uint8)

    return city_map

""" ENTITIES """
class Bot:
    def __init__(self, pos: tuple, lifetime_min, lifetime_max):
        self.pos = pos  # (x, y)
        self.prev_pos = pos
        self.next_pos = None
        self.steps = 0
        self.alive = True
        self.first_step = True
        self.lifetime = random.randint(lifetime_min, lifetime_max)

class Agent:
    def __init__(self, start_pos: tuple):
        self.pos = start_pos  # (x, y)
        self.prev_pos = self.pos
        self.steps = 0
        self.alive = True

class Environment:
    def __init__(
        self,
        map_path: str,
        flat_obs: bool = False,
        bot_spawn_chance: float = 0.02,
        bots_lifetime_min: int = 15,
        bots_lifetime_max: int = 150,
        agent_finish_reward = 100,
        agent_lose_reward = -100,
        step_reward = -1, # Agent reward for valid step
        wall_penalty = -5, # Agent reward for invalid step (into wall)
        max_steps: int = 500, # Max number of agent steps
        n_vision: int = 1, # How many cells in each direction agents see
        human_agent: str|None = None,

        render_mode: str | None = None,
        sim_speed: int = 10,
        cell_size: int = 20,
        bot_color: tuple = (0, 0, 0),
        agent_color: tuple = (255, 0, 0),
        map_config_path = None,
        seed = None
    ):
        # System
        self.city_map = prepare_map(map_path)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.bot_spawns = (np.argwhere(self.city_map == 3).astype(int)) # [(y, x), ...]
        self.bots_lifetime_min = bots_lifetime_min
        self.bots_lifetime_max = bots_lifetime_max
        self.bots: list[Bot] = []

        self.agent_spawns = [(int(f[1]), int(f[0])) for f in np.argwhere(self.city_map == 2)]
        self.agent_finishes = [(int(f[1]), int(f[0])) for f in np.argwhere(self.city_map == 4)]
        self.map_config_path = map_config_path

        if len(self.agent_spawns) != len(self.agent_finishes):
            raise ValueError(f"Number of spawns ({len(self.agent_spawns)}) does "
                             f"not match number of finishes ({len(self.agent_finishes)})")

        self.agents: dict = {} # {agent_id: 'obj': ..., 'spawn': ..., 'finish_pos': ...}
        self._assign_finishes()

        # Environment settings
        self.flat_obs = flat_obs
        self.bot_spawn_chance = bot_spawn_chance
        self.agent_finish_reward = agent_finish_reward
        self.agent_lose_reward = agent_lose_reward
        self.step_reward = step_reward
        self.wall_penalty = wall_penalty
        self.max_steps = max_steps
        self.n_vision = n_vision

        if human_agent and render_mode != 'human':
            raise ValueError("human_agent requires render_mode='human'")

        self.human_agent = human_agent
        self._human_action = None

        # Visual
        self.render_mode = render_mode
        self.sim_speed = sim_speed
        self.cell_size = cell_size
        self.cell_colors: dict|None = None

        self.map_img = None
        self.window = None
        self.clock = None

        self.bot_img = None
        self.agent_imgs: dict = {}
        self.agent_dead_imgs: dict = {}
        self.bot_color = bot_color
        self.agent_color = agent_color
        self.font = None
        self.spawn_text_color = (0, 0, 0)
        self.finish_text_color = (255, 255, 255)

        if render_mode == 'human':
            pygame.init()
            # Map
            map_h, map_w = self.city_map.shape
            window_size = (map_w * self.cell_size, map_h * self.cell_size)

            self.cell_colors = {
                0: (200, 200, 200),
                1: (29, 95, 153),
                2: (85, 255, 255),
                3: (255, 241, 82),
                4: (255, 0, 0)
            }

            map_surf = pygame.Surface((map_w * self.cell_size, map_h * self.cell_size))

            for y, row in enumerate(self.city_map):
                for x, cell in enumerate(row):
                    pygame.draw.rect(
                        map_surf,
                        self.cell_colors[cell],
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )

            self.map_img = map_surf
            self.window = pygame.display.set_mode(window_size)
            self.clock = pygame.time.Clock()

            # Font
            self.font = pygame.font.SysFont(None, self.cell_size)

            # Entity images (templates)
            self.bot_img = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.circle(
                self.bot_img,
                self.bot_color,
                (self.cell_size // 2, self.cell_size // 2),
                self.cell_size // 2
            )

            # Image caching (fewer calculations in loop)
            for agent_id in self.agents:
                # The base itself
                agent_img = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                pygame.draw.circle(
                    agent_img,
                    self.agent_color,
                    (self.cell_size // 2, self.cell_size // 2),
                    self.cell_size // 2
                )

                # Text (number)
                id_num = agent_id.split('agent_')[1]
                text = self.font.render(id_num, True, (255, 255, 255))
                text_rect = text.get_rect(center=(self.cell_size // 2, self.cell_size // 2))
                agent_img.blit(text, text_rect)
                self.agent_imgs[agent_id] = agent_img

                # Semi-transparent images for dead ones
                agent_img_dead = agent_img.copy()
                agent_img_dead.set_alpha(70)
                self.agent_dead_imgs[agent_id] = agent_img_dead

    """--- Helper methods for bots and agent(s) ---"""
    def _get_directions(self, pos: tuple, n_vision=1):
        x, y = pos
        directions = []
        for i in range(1, n_vision + 1):
            directions.append((x, y - i)) # top
            directions.append((x, y + i)) # bottom
            directions.append((x - i, y)) # left
            directions.append((x + i, y)) # right

        return directions

    def _get_valid_paths(self, pos: tuple):
        valid_paths = []

        for neighbor in self._get_directions(pos):
            nc_x, nc_y = neighbor

            if (0 <= nc_y < self.city_map.shape[0]) and (0 <= nc_x < self.city_map.shape[1]):
                if self.city_map[nc_y, nc_x] != 1:
                    valid_paths.append((nc_x, nc_y))

        return valid_paths

    """For bots"""
    def _move_bots(self):

        for bot in self.bots:
            if not bot.alive: continue

            if bot.lifetime and bot.steps == bot.lifetime:
                bot.alive = False
                continue

            if bot.first_step:
                bot.first_step = False
                continue

            # 0 - up, 1 - down, 2 - left, 3 - right
            valid_paths = self._get_valid_paths(bot.pos)

            bot.steps += 1

            # Dead end
            if len(valid_paths) == 1:
                bot.prev_pos = bot.pos
                bot.pos = valid_paths[0]
                continue

            if bot.next_pos:
                bot.prev_pos = bot.pos
                bot.pos = bot.next_pos
                bot.next_pos = None
                continue

            # Fork
            # 10% - stop, next step - opposite direction
            if random.randint(1, 10) == 1:
                bot.next_pos = bot.prev_pos
            else:
                if bot.prev_pos in valid_paths:
                    valid_paths.remove(bot.prev_pos)
                bot.prev_pos = bot.pos
                # Random choice from remaining options
                bot.pos = random.choice(valid_paths)

    """For agents"""
    # Link finishes to agents (randomly)
    def _assign_finishes(self):
        if self.map_config_path:
            try:
                with open(self.map_config_path, 'r') as f:
                    config_data = json.load(f)
                for a_id, a_cfg in enumerate(config_data['agents']):
                    self.agents[f'agent_{a_id}'] = {
                        'obj': Agent(tuple(a_cfg['spawn'])),
                        'spawn': tuple(a_cfg['spawn']),
                        'finish': tuple(a_cfg['finish'])
                    }
                return
            except (json.decoder.JSONDecodeError, TypeError, KeyError):
                logger.error(f"Invalid config file: {self.map_config_path} - falling back to default config")

        for a_id, spawn, finish in zip(
                range(len(self.agent_spawns)),
                self.agent_spawns,
                self.agent_finishes
        ):
            self.agents[f'agent_{a_id}'] = {'obj': Agent(spawn), 'spawn': spawn, 'finish': finish}

    def _apply_action(self, agent: Agent, action) -> bool:
        """
        action:
        0 - вверх
        1 - вниз
        2 - влево
        3 - вправо
        4 - стоять
        """

        agent.prev_pos = agent.pos

        if action == 4:
            return True

        t_x, t_y = self._get_directions(agent.pos)[action]
        if (
            (0 <= t_y < self.city_map.shape[0]) and
            (0 <= t_x < self.city_map.shape[1]) and
            self.city_map[t_y, t_x] != 1 # not a wall
        ):
            agent.pos = (t_x, t_y)
            return True
        else:
            return False

    def _get_obs(self, agent: Agent, finish: tuple):
        # Bots nearby (n-cells)
        bot_positions = set(bot.pos for bot in self.bots)
        near_bots = tuple(1 if pos in bot_positions else 0
                          for pos in self._get_directions(agent.pos, self.n_vision))

        # Other alive agents nearby (n-cells)
        agents_positions = set(agent_data['obj'].pos
                               for _, agent_data in self.agents.items()
                               if agent_data['obj'] != agent and agent_data['obj'].alive)
        near_agents = tuple(1 if pos in agents_positions else 0
                            for pos in self._get_directions(agent.pos, self.n_vision))

        # Walls nearby (4 cells nearby)
        valid_cells = self._get_valid_paths(agent.pos)
        near_walls = tuple(0 if pos in valid_cells else 1
                          for pos in self._get_directions(agent.pos, 1))

        # Normalized distance to finish
        a_x, a_y = agent.pos
        f_x, f_y = finish

        finish_dir = (
            (f_x - a_x) / self.city_map.shape[1],
            (f_y - a_y) / self.city_map.shape[0]
        )

        obs = near_bots, near_agents, near_walls, finish_dir
        return obs if not self.flat_obs else np.concatenate(obs)

    # Manual control
    def _get_human_action(self):
        self._human_action = None
        while self._human_action is None:
            self._render()

        return self._human_action

    """Environment base"""
    @staticmethod
    def action_sample(self):
        return random.randint(0, 3)

    def step(self, actions: dict[str, int]):

        results: dict[str, dict] = {
            agent_id: {'obs': None,'reward': 0,'terminated': False,'truncated': False, 'info': {}}
            for agent_id in actions
        }

        # reward - Reward
        # terminated - Has agent reached terminal state (Finish/Accident)
        # truncated - Has agent step limit ended
        # info - Additional information

        # Bot spawn
        for spawn in self.bot_spawns:
            if random.random() < self.bot_spawn_chance:
                s_y, s_x = spawn
                self.bots.append(Bot((s_x, s_y),
                                     self.bots_lifetime_min, self.bots_lifetime_max))

        # Manual control
        if self.human_agent and self.human_agent in actions:
            actions[self.human_agent] = self._get_human_action()

        # Is agent step successful (position change or hit a wall)
        for agent_id, action in actions.items():
            if not self.agents[agent_id]['obj'].alive:
                results[agent_id]['terminated'] = True
                continue

            is_action_valid = self._apply_action(self.agents[agent_id]['obj'], action) # agent by id
            # Agent reward for step (or wall penalty)
            results[agent_id]['reward'] = self.step_reward if is_action_valid else self.wall_penalty

        for agent_id in actions:
            if not self.agents[agent_id]['obj'].alive:
                continue

            agent = self.agents[agent_id]['obj']
            agent.steps += 1 # Agent took a step
            # GOOOOOAAL
            if agent.pos == self.agents[agent_id]['finish']:
                results[agent_id]['reward'] = self.agent_finish_reward
                results[agent_id]['terminated'] = True
                agent.alive = False
            # Not goal...
            if agent.steps >= self.max_steps:
                results[agent_id]['truncated'] = True

        # Bots - step
        self._move_bots()

        # Accidents
        accidents = {}  # (x, y): [...]
        swaps = {}  # ((x1, y1),(x2, y2)): [...] -> (0,1)(0,2) = (0,2)(0,1)
        to_kill = set()

        entities = self.bots + [
            self.agents[agent_id]['obj'] for agent_id in actions.keys()
            if self.agents[agent_id]['obj'].alive
        ]

        for entity in entities:
            # Has bot lifetime passed
            if not entity.alive:
                to_kill.add(entity)
                continue

            # If key exists - add bot to value (list), if not - create key and add bot there
            accidents.setdefault(entity.pos, []).append(entity)
            # note: sorted returns list, so wrapped in tuple
            swaps.setdefault(tuple(sorted((entity.pos, entity.prev_pos))), []).append(entity)

        for accident in accidents.values():
            if len(accident) > 1:
                for entity in accident:
                    to_kill.add(entity)

        for swap in swaps.values():
            if len(swap) > 1:
                for entity in swap:
                    to_kill.add(entity)

        for agent_id in actions:
            agent = self.agents[agent_id]['obj']
            a_finish = self.agents[agent_id]['finish']
            results[agent_id]['obs'] = self._get_obs(agent, a_finish)
            results[agent_id]['info'] = {
                'steps': agent.steps,
                'pos': agent.pos,
                'manhattan': abs(agent.pos[0] - a_finish[0]) + abs(agent.pos[1] - a_finish[1]),
                'finish': a_finish
            }

            # Check agent finish or collision
            if agent in to_kill and results[agent_id]['reward'] != self.agent_finish_reward:
                # Agent is dead
                self.agents[agent_id]['obj'].alive = False

                results[agent_id]['terminated'] = True
                results[agent_id]['reward'] = self.agent_lose_reward

        # Cleanup of dead bots
        self.bots = [bot for bot in self.bots if bot not in to_kill]

        # Environment state (for all agents - separate dictionaries)
        obs, rewards, terminated, truncated, info = ({} for _ in range(5))

        for agent_id in results:
            obs[agent_id] = results[agent_id]['obs']
            rewards[agent_id] = results[agent_id]['reward']
            terminated[agent_id] = results[agent_id]['terminated']
            truncated[agent_id] = results[agent_id]['truncated']
            info[agent_id] = results[agent_id]['info']

        self._render()
        return obs, rewards, terminated, truncated, info

    def _render(self):
        if not self.window: return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and self.human_agent:
                if event.key == pygame.K_UP: self._human_action = 0
                if event.key == pygame.K_DOWN: self._human_action = 1
                if event.key == pygame.K_LEFT: self._human_action = 2
                if event.key == pygame.K_RIGHT: self._human_action = 3
                if event.key == pygame.K_SPACE: self._human_action = 4

        self.clock.tick(self.sim_speed)
        self.window.blit(self.map_img, (0, 0))

        for bot in self.bots:
            self.window.blit(self.bot_img, (bot.pos[0] * self.cell_size, bot.pos[1] * self.cell_size))

        for agent_id in self.agents:
            agent = self.agents[agent_id]['obj']

            if not agent.alive:
                # Image of dead agent
                self.window.blit(
                    self.agent_dead_imgs[agent_id],
                    (agent.pos[0] * self.cell_size, agent.pos[1] * self.cell_size)
                )
            else:
                # Regular image
                self.window.blit(
                    self.agent_imgs[agent_id],
                    (agent.pos[0] * self.cell_size, agent.pos[1] * self.cell_size)
                )

            # Starts
            id_num = agent_id.split('agent_')[1]

            spawn = self.agents[agent_id]['spawn']
            text = self.font.render(id_num, True, self.spawn_text_color)
            text_rect = text.get_rect(center=(
                spawn[0] * self.cell_size + self.cell_size // 2,
                spawn[1] * self.cell_size + self.cell_size // 2,
            ))
            self.window.blit(text, text_rect)

            # Finishes
            finish = self.agents[agent_id]['finish']
            text = self.font.render(id_num, True, self.finish_text_color)
            text_rect = text.get_rect(center=(
                finish[0] * self.cell_size + self.cell_size // 2,
                finish[1] * self.cell_size + self.cell_size // 2,
            ))
            self.window.blit(text, text_rect)

        pygame.display.flip()

    def reset(self, seed=None, randomize_spawns=False, randomize_finishes=False):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        obs = {}
        if randomize_spawns:
            for agent_id, spawn in zip(
                    self.agents, random.sample(self.agent_spawns, len(self.agent_spawns))
            ):
                self.agents[agent_id]['spawn'] = spawn

        if randomize_finishes:
            for agent_id, finish in zip(
                    self.agents, random.sample(self.agent_finishes,len(self.agent_finishes))
            ):
                self.agents[agent_id]['finish'] = finish

        # Reset agent positions and parameters
        for agent_id in self.agents:
            agent = self.agents[agent_id]['obj']

            agent.alive = True # Agent is alive
            agent.steps = 0
            agent.pos = self.agents[agent_id]['spawn']
            agent.prev_pos = agent.pos
            obs[agent_id] = self._get_obs(agent, self.agents[agent_id]['finish'])

        self.bots = []
        return obs

""" ------------------ PARALLELISM ------------------ """

def _worker_fn(pipe, env_fn, seed=None):
    # Reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = env_fn()
    while True:
        cmd, data = pipe.recv()
        if cmd == 'reset':
            obs = env.reset(**data)
            pipe.send(obs)
        elif cmd == 'step':
            obs, reward, terminated, truncated, info = env.step(data)
            if all(terminated[a] or truncated[a] for a in terminated):
                # Saving terminal observation for non-QL algorithms
                for agent_id in obs:
                    info[agent_id]['terminated_obs'] = obs[agent_id]
                obs = env.reset()

            res = obs, reward, terminated, truncated, info
            pipe.send(res)
        elif cmd == 'close':
            break

    pipe.close()

class ParallelEnvironment:
    def __init__(self, n_envs: int, map_path: str, seed=None, **env_kwargs):
        self._pipes = [] # All connections with workers
        self._workers: list[mp.Process] = []

        # render_mode is always None for parallelism
        if env_kwargs.pop('render_mode', None) is not None:
            logger.warning('The render_mode parameter is ignored in parallel mode')

        # human_agent is also None for parallelism
        if env_kwargs.pop('human_agent', None) is not None:
            logger.warning('The human_agent parameter is ignored in parallel mode')

        # Template environment for reading (agent ids, settings, etc.)
        self.env = Environment(map_path, render_mode=None, seed=seed, **env_kwargs)

        # Factory for Environment objects
        env_fn = partial(Environment, map_path=map_path, render_mode=None, **env_kwargs)

        for i in range(n_envs):
            w_seed = seed + i if seed is not None else None
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(
                target=_worker_fn,
                args=(child_conn, env_fn, w_seed),
                daemon=True # workers live only while main process is alive
            )
            proc.start()
            child_conn.close()

            self._pipes.append(parent_conn)
            self._workers.append(proc)

    def reset(self, **kwargs):
        """ obs[i] = {agent_id, obs} for i-th env """
        for pipe in self._pipes:
            pipe.send(('reset', kwargs))

        return [pipe.recv() for pipe in self._pipes]

    def step(self, actions):
        """ actions[i] = {agent_id, action} for i-th env """
        for pipe, acts in zip(self._pipes, actions):
            pipe.send(('step', acts))

        results = [pipe.recv() for pipe in self._pipes]
        return results

    def close(self):
        for pipe in self._pipes:
            pipe.send(('close', None))

        # Wait until all workers close
        for worker in self._workers:
            worker.join()
