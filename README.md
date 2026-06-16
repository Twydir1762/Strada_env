# Strada-env
<img width="1925" height="735" alt="DEMONSTATION" src="https://github.com/user-attachments/assets/6c455cec-98f9-471c-9369-065ba4daf939" />

A multi-agent grid-based environment for reinforcement learning. Agents navigate a city map toward their individual finish points while avoiding randomly spawning bots. Built with pygame, follows a PettingZoo-style step API.

## Installation

```bash
pip install git+https://github.com/Twydir1762/Strada_env.git
```

## Environment

The map is a 2D grid where each cell has a type:

| Value | Type |
|---|---|
| `0` | Road |
| `1` | Wall |
| `2` | Agent spawn |
| `3` | Bot spawn |
| `4` | Agent finish |

```
1 1 1 1 1 1 1 1 1 1 1 1
1 4 0 1 1 0 0 0 0 0 0 1
1 0 0 0 0 0 1 1 1 1 0 1
1 0 1 1 1 1 1 0 0 1 0 1
1 0 0 0 0 1 0 0 0 1 0 1
1 1 1 1 0 1 0 1 0 1 0 1
1 0 0 0 0 0 0 1 0 0 0 1
1 3 1 1 0 1 1 1 1 1 0 1
1 0 0 0 0 0 0 0 0 0 0 1
1 1 1 1 1 1 1 1 1 0 1 1
1 2 0 0 0 0 0 0 0 0 0 1
1 1 1 1 1 1 1 1 1 1 1 1
```

Maps are plain `.txt` files and can be created with the built-in [map editor](#map-editor).

```python
from strada_env import Environment
 
env = Environment(map_path='maps/12x12_1a.txt', bot_spawn_chance=0.2, ...)
 
obs = env.reset()  # {agent_id: observation, ...}
 
actions = {agent_id: env.action_sample() for agent_id in obs}
obs, rewards, terminated, truncated, info = env.step(actions)
```
 
**Observation** per agent is a tuple of arrays: `(near_bots, near_agents, near_walls, finish_dir)`.
 
**Actions:** `0` up · `1` down · `2` left · `3` right · `4` stay.
 
### Parameters
 
**Bots**
 
| Parameter | Default | Description |
|---|---|---|
| `bot_spawn_chance` | `0.02` | Probability of spawning a bot each step at each spawn point |
| `bots_lifetime_min` | `15` | Minimum steps a bot lives before despawning |
| `bots_lifetime_max` | `150` | Maximum steps a bot lives before despawning |
 
**Rewards**
 
| Parameter | Default | Description |
|---|---|---|
| `step_reward` | `-1` | Reward for each valid step |
| `wall_penalty` | `-5` | Penalty for stepping into a wall |
| `agent_finish_reward` | `100` | Reward for reaching the finish |
| `agent_lose_reward` | `-100` | Reward for colliding with a bot or other agent |
 
**Episode**
 
| Parameter | Default | Description |
|---|---|---|
| `max_steps` | `500` | Max steps per episode before truncation |
 
**Observation**
 
| Parameter | Default | Description |
|---|---|---|
| `n_vision` | `1` | Cells visible in each direction |
| `flat_obs` | `False` | Flatten the observation tuple into a single array |

### Parallel training

```python
from strada_env import ParallelEnvironment

p_env = ParallelEnvironment(n_envs=6, map_path='maps/30x30_2a.txt', **env_kwargs)

all_obs = p_env.reset()         # list of obs dicts, one per env
results = p_env.step(actions)   # list of (obs, reward, terminated, truncated, info)
p_env.close()
```

When an episode ends in a parallel env, the environment auto-resets and stores the terminal observation in `info[agent_id]['terminated_obs']`.

### Human agent

Pass `human_agent='agent_0'` to control one agent with arrow keys (space = stay).

## Examples

Ready-to-run Q-learning scripts are in `examples/`:

| Script | Description |
|---|---|
| `one_env_q-learning.py` | Single environment, multi-agent Q-learning |
| `parallel_q-learning.py` | Parallel environments for faster training |

Both scripts include `train` and `test` functions with configurable hyperparameters at the top of the file.

## Map Editor

<img width="1000" height="563" alt="Map_editor_demo" src="https://github.com/user-attachments/assets/e135dc50-1b7d-4bda-8353-c85fd6397c9d" />


Create and edit maps visually:

```bash
strada map-editor
strada map-editor --window-size 1000 1000 --map-size 30 30
```

Maps are saved as plain-text `.txt` files and passed directly to `Environment(map_path=...)`.\
Ready-made examples are in the `maps/` folder.
