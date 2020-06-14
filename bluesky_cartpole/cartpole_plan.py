from collections import deque

import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps

from bluesky_cartpole.cartpole import CartPole, get_cartpole_agent


# logging.getLogger("bluesky").setLevel("DEBUG")
# In [8]: logging.basicConfig()
def train_agent(env, agent, max_episodes, *, md=None, next_point_callback=None):
    md = md or {}

    queue = deque()
    episode_count = 1
    total_reward = 0.0

    # this function will be called with every document:
    #   run_start, descriptor, event, event, ..., event, stop
    def get_next_point_callback(name, doc):
        nonlocal episode_count
        nonlocal total_reward

        if name == "event" and episode_count <= max_episodes:
            states, terminal, reward, state_after_reset = doc["data"]["cartpole"]
            total_reward += reward
            print(f"terminal: {terminal}")
            agent.observe(reward=reward, terminal=terminal)
            if terminal == 1:
                print(f"end of episode {episode_count}")
                print(f"total reward: {total_reward}")
                total_reward = 0.0
                episode_count += 1
                states = state_after_reset

            action = agent.act(states=states)
            queue.append(action)

    if next_point_callback is None:
        next_point_callback = get_next_point_callback

    @bpp.subs_decorator(next_point_callback)
    @bpp.run_decorator(md=md)
    def rl_training_plan():
        uids = []

        state = env.cartpole_env.reset()
        action = agent.act(states=state)
        queue.append(action)

        while len(queue) > 0:
            action = queue.pop()
            # set the action Signal to the next action
            yield from bps.mv(env.next_action, action)
            # execute the action
            uid = yield from bps.trigger_and_read([env])
            uids.append(uid)

        return uids

    return (yield from rl_training_plan())


def train_cartpole_agent():
    print("don't forget to start tensorboard: tensorboard --log-dir data")
    cartpole = CartPole()
    cartpole_agent = get_cartpole_agent(cartpole)

    yield from train_agent(env=cartpole, agent=cartpole_agent, max_episodes=100)
