from strada_env import Environment, ParallelEnvironment
from examples.controllers import MultiAgentController
from tqdm import tqdm
import sys


MAP_PATH = 'maps/30x30_2a.txt'
Q_TABLE = 'q_table_30x30_2a.pickle'

# TRAIN SETTINGS
TRAIN_EPISODES = 10000
TEST_EPISODES = 100
EPS = 0.5
MIN_EPS = 0.01
EPS_DECAY = 0.9995
N_ENVS = 6

# ENV SETTINGS
MAX_STEPS = 300
BOT_SPAWN_CHANCE = 0.2
BOTS_LIFETIME = (20, 80)
STEP_REWARD = -2
AGENT_FINISH_REWARD = 100
AGENT_LOSE_REWARD = -100

# TEST SETTINGS
FPS = 5


def ql_mult_train_p(episodes, q_table_path, map_path, n_envs, seed=None):
    p_env = ParallelEnvironment(
        n_envs=n_envs,
        map_path=map_path,
        bot_spawn_chance = BOT_SPAWN_CHANCE,
        agent_finish_reward = AGENT_FINISH_REWARD,
        agent_lose_reward = AGENT_LOSE_REWARD,
        step_reward = STEP_REWARD,
        max_steps = MAX_STEPS,
        bots_lifetime_min = BOTS_LIFETIME[0],
        bots_lifetime_max = BOTS_LIFETIME[1],
        seed=seed
    )

    agents_ids = list(p_env.env.agents.keys())

    # Initial setup
    mcon = MultiAgentController(
        agent_ids=agents_ids,
        eps = EPS,
        min_eps = MIN_EPS,
        eps_decay = EPS_DECAY,
        lr = 0.1,
        gamma = 0.99
    )

    all_obs = p_env.reset() # [obs0, obs1, ...]
    states = [{a_id: obs[a_id] for a_id in agents_ids} for obs in all_obs]
    active_agents = [set(agents_ids) for _ in range(n_envs)]

    completed_ep = 0
    pbar = tqdm(total=episodes, desc="Training", file=sys.stdout)

    while completed_ep < episodes:
        # Action selection for all agents (in each environment)
        agents_actions = [
            {a_id: mcon.controllers[a_id].choose_action(states[i][a_id])
             for a_id in sorted(active_agents[i])} for i in range(n_envs)
        ]

        # Step for all environments (parallel)
        results = p_env.step(agents_actions)

        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            # Update agents' Q-table
            for a_id in agents_actions[i]:
                next_state = obs[a_id]
                agent_done = terminated[a_id] or truncated[a_id]

                mcon.controllers[a_id].update_q(
                    states[i][a_id],
                    next_state,
                    agents_actions[i][a_id],
                    reward[a_id],
                    agent_done
                )

                if agent_done:
                    active_agents[i].discard(a_id)

            if any('terminated_obs' in info[a_id] for a_id in info):
                completed_ep += 1
                pbar.update(1)
                active_agents[i] = set(agents_ids)
                states[i] = {a_id: obs[a_id] for a_id in agents_ids}

            else:
                for a_id in sorted(active_agents[i]):
                    states[i][a_id] = obs[a_id]

        # Update eps
        eps = max(MIN_EPS, EPS * (EPS_DECAY ** completed_ep))
        for a_id in agents_ids:
            mcon.controllers[a_id].set_eps(eps)

        # Average number of states
        avg_states = sum(len(c.q_table) for c in mcon.controllers.values()) / len(mcon.controllers)
        pbar.set_postfix(eps=f"{eps:.3f}", states=int(avg_states))

    pbar.close()
    p_env.close()
    mcon.save_qs(q_table_path)

def ql_mult_test(episodes, q_table_path, map_path, render_mode, seed=None, human_agent=None):
    env = Environment(
        map_path=map_path,
        cell_size=35,
        sim_speed=FPS,
        render_mode=render_mode,
        bot_spawn_chance=BOT_SPAWN_CHANCE,
        agent_finish_reward=AGENT_FINISH_REWARD,
        agent_lose_reward=AGENT_LOSE_REWARD,
        step_reward=STEP_REWARD,
        max_steps=MAX_STEPS,
        bots_lifetime_min=BOTS_LIFETIME[0],
        bots_lifetime_max=BOTS_LIFETIME[1],
        human_agent=human_agent,
        seed=seed
    )

    # Initial setup
    mcon = MultiAgentController(
        agent_ids=env.agents.keys(),
        eps = 0
    )

    mcon.load_qs(q_table_path)

    wins = {a_id: 0 for a_id in env.agents} # {a_id: int}

    for _ in range(episodes):
        obs = env.reset()
        # Revive all agents
        active_agents = set(env.agents)

        while active_agents:
            agent_actions = {a_id: 0 for a_id in sorted(active_agents)}
            states = {} # states for all agents per step

            # Action selection for all agents
            for a_id in agent_actions:
                states[a_id] = obs[a_id]
                agent_actions[a_id] = mcon.controllers[a_id].choose_action(states[a_id])

            obs, reward, terminated, truncated, info = env.step(agent_actions)

            # Clean up corpses
            for a_id in agent_actions:
                agent_done = terminated[a_id] or truncated[a_id]
                if agent_done:
                    if reward[a_id] == env.agent_finish_reward:
                        wins[a_id] += 1
                    active_agents.discard(a_id)

    return wins

if __name__ == '__main__':
   ql_mult_train_p(TRAIN_EPISODES, Q_TABLE, MAP_PATH, N_ENVS)
   ql_mult_test(TEST_EPISODES, Q_TABLE, MAP_PATH, 'human')
