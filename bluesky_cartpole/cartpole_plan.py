from collections import deque
import pprint

import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps

from bluesky_cartpole.cartpole import CartPole, get_cartpole_agent


# logging.getLogger("bluesky").setLevel("DEBUG")
# In [8]: logging.basicConfig()
def train_agent(env, agent, episode_count, *, md=None, next_point_callback=None):
    md = md or {}

    queue = deque()
    episode_i = 1
    total_reward = 0.0

    # this function will be called with every document:
    #   run_start, descriptor, event, event, ..., event, stop
    def get_next_point_callback(name, doc):
        nonlocal episode_i
        nonlocal total_reward

        if name == "event" and episode_i <= episode_count:
            states = doc["data"]["cartpole_next_state"]
            terminal = doc["data"]["cartpole_terminal"]
            reward = doc["data"]["cartpole_reward"]
            state_after_reset = doc["data"]["cartpole_state_after_reset"]

            total_reward += reward
            agent.observe(reward=reward, terminal=terminal)
            if terminal == 1:
                total_reward = 0.0
                episode_i += 1
                states = state_after_reset

            action = agent.act(states=states)
            queue.append(action)
        else:
            # the agent is not interested in this document
            pass

    if next_point_callback is None:
        next_point_callback = get_next_point_callback

    @bpp.subs_decorator(next_point_callback)
    @bpp.run_decorator(md=md)
    @bpp.stage_decorator(devices=[env])
    def rl_training_plan():
        uids = []

        # staging the cartpole device resets its state
        state_after_reset = env.state_after_reset.get()
        action = agent.act(states=state_after_reset)
        queue.append(action)

        while len(queue) > 0:
            action = queue.pop()
            # set the action Signal to the next action
            yield from bps.mv(env.action, action)
            # execute the action
            uid = yield from bps.trigger_and_read([env])
            uids.append(uid)

        return uids

    return (yield from rl_training_plan())


def train_cartpole_agent(episode_count):
    print("don't forget to start tensorboard: tensorboard --log-dir data")
    cartpole = CartPole()
    cartpole_agent = get_cartpole_agent(cartpole)

    yield from train_agent(env=cartpole, agent=cartpole_agent, episode_count=episode_count)
