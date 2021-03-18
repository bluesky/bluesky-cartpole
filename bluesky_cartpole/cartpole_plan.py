from collections import deque

import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps

from bluesky_cartpole.cartpole import CartPole, get_cartpole_agent


# logging.getLogger("bluesky").setLevel("DEBUG")
# In [8]: logging.basicConfig()
def train_agent(env_device, agent, episode_count, *, md=None, next_point_callback=None):
    """
    A bluesky "plan" that trains an agent to play cartpole.

    Parameters
    ----------
    env_device: bluesky_cartpole.cartpole.CartPole
        the cartpole training environment
    agent: a Tensorforce Agent
        the agent that will be trained to play cartpole
    episode_count: int
        number of training episodes
    md: dict, optional
        bluesky metadata dictionary
    next_point_callback: function(name, doc), optional
        a function taking a bluesky name, document pair

    Return
    ------
    no return value
    """
    if md is None:
        md = {}

    queue = deque()
    episode_i = 1
    step_i = 0
    total_reward = 0.0

    # this function will be subscribed to the RunEngine
    # it will be called with every document:
    #   run_start, descriptor, event, event, ..., event, stop
    # only event documents have information relevant to the agent
    def get_next_point_callback(name, doc):
        nonlocal episode_i
        nonlocal step_i
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
                step_i = 0
                states = state_after_reset
            else:
                step_i += 1

            action = agent.act(states=states)
            queue.append(action)
        else:
            # the agent is not interested in this document
            pass

    if next_point_callback is None:
        next_point_callback = get_next_point_callback

    @bpp.subs_decorator(next_point_callback)
    @bpp.run_decorator(md=md)
    @bpp.stage_decorator(devices=[env_device])
    def rl_training_plan():
        nonlocal episode_i
        nonlocal step_i

        uids = []

        # staging the cartpole device resets its state
        state_after_reset = env_device.state_after_reset.get()
        action = agent.act(states=state_after_reset)
        queue.append(action)

        while len(queue) > 0:
            action = queue.pop()

            # take a break for evaluation
            if episode_i % 10 == 0 and step_i == 0:
                print(f"time for evaluation: episode_i: {episode_i}")
                sum_rewards = 0.0
                evaluation_episode_count = 100
                for _ in range(evaluation_episode_count):
                    states = env_device.cartpole_env.reset()
                    internals = agent.initial_internals()
                    terminal = False
                    while not terminal:
                        actions, internals = agent.act(
                            states=states,
                            internals=internals,
                            independent=True,
                            deterministic=True,
                        )
                        states, terminal, reward = env_device.cartpole_env.execute(
                            actions=actions
                        )
                        sum_rewards += reward
                env_device.cartpole_env.reset()
                average_evaluation_reward = sum_rewards / evaluation_episode_count
                # this will go in the next event document
                yield from bps.mv(
                    env_device.average_evaluation_reward, average_evaluation_reward
                )

            # set the action Signal to the next action
            yield from bps.mv(env_device.action, action)
            # execute the action
            uid = yield from bps.trigger_and_read([env_device])
            uids.append(uid)

            # clear the average evaluation reward for the next event
            # if episode_i % 10 == 0 and step_i == 1:
            yield from bps.mv(env_device.average_evaluation_reward, 0.0)

        return uids

    return (yield from rl_training_plan())


def train_cartpole_agent(agent_name, episode_count):
    print("don't forget to start tensorboard: tensorboard --log-dir data")

    cartpole_device = CartPole()
    cartpole_agent, agent_parameters = get_cartpole_agent(
        agent_name=agent_name, cartpole_device=cartpole_device
    )

    md = {
        "agent_name": agent_name,
        "episode_count": episode_count,
        "agent_parameters": agent_parameters,
    }

    yield from train_agent(
        env_device=cartpole_device,
        agent=cartpole_agent,
        episode_count=episode_count,
        md=md,
    )
