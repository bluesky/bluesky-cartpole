import math

import numpy as np

from tensorforce import Environment
from tensorforce.agents import Agent

from ophyd import Component as Cpt, Device, Signal
from ophyd.status import Status


class CartPole(Device):
    # read actions from this Signal
    # actions are set by an agent outside this device
    action = Cpt(Signal, value=0)

    next_state = Cpt(Signal, value=np.asarray([math.nan, math.nan, math.nan, math.nan]))
    reward = Cpt(Signal, value=0.0)
    terminal = Cpt(Signal, value=0)
    state_after_reset = Cpt(
        Signal, value=np.asarray([math.nan, math.nan, math.nan, math.nan])
    )

    average_evaluation_reward = Cpt(Signal, value=0.0)

    def __init__(self, name="cartpole", prefix="CARTPOLE", **kwargs):
        super().__init__(name=name, prefix=prefix)

        self.cartpole_env = Environment.create(
            environment="gym", level="CartPole-v1", **kwargs
        )

    def stage(self):
        state_after_reset_ = self.cartpole_env.reset()
        self.state_after_reset.put(state_after_reset_)
        return [self]

    def trigger(self):
        """
        Read the next action from the self.action signal and
        execute that action in the cartpole environment. Record
        the new state of the environment (_next_state), if the
        cartpole episode has ended (_terminal), and the reward
        for the action (_reward).

        Returns
        -------
        status_finished: Status
            a status object in the `finished` state
        """

        _next_state, _terminal, _reward = self.cartpole_env.execute(
            actions=self.action.get()
        )

        self.next_state.put(_next_state)
        self.terminal.put(_terminal)
        self.reward.put(_reward)

        # self.cartpole_env indicates two different terminal conditions:
        #    terminal==1 -- the pole fell over
        #    terminal==2 -- the maximum number of timesteps have been taken
        if self.terminal.get() > 0:
            self.state_after_reset.put(self.cartpole_env.reset())
        else:
            self.state_after_reset.put(
                np.asarray([math.nan, math.nan, math.nan, math.nan])
            )

        status_finished = Status()
        status_finished.set_finished()
        return status_finished

    def unstage(self):
        return [self]


def get_cartpole_agent(agent_name, cartpole_device):
    if agent_name == "a2c":
        agent_parameters = dict(
            agent=agent_name,
            batch_size=11,
            variable_noise=0.1,
            l2_regularization=0.05,  # does this help with catastrophic forgetting?
            horizon=10,  # 10 is good, 1 is bad, 5 is bad, 20 is ok, 15 is bad
            summarizer=dict(
                directory="data/summaries/" + agent_name,
                # list of labels, or 'all'
                labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
                frequency=10,  # store values every 10 timesteps
            ),
        )
        agent = Agent.create(
            # agent="a2c",
            environment=cartpole_device.cartpole_env,
            # the gym cartpole environment will supply max_episode_timesteps
            # max_episode_timesteps=max_turns,
            **agent_parameters,
        )
    elif agent_name == "ppo":
        agent_parameters = dict(batch_size=10, variable_noise=0.1,)
        agent = Agent.create(
            # agent="ppo",
            environment=cartpole_device.cartpole_env,
            **agent_parameters,
        )
    elif agent_name == "dqn":
        agent_parameters = dict(batch_size=100, variable_noise=0.2, memory=1000)

        agent = Agent.create(
            # agent="dqn",
            environment=cartpole_device.cartpole_env,
            # memory=1000,
            # batch_size=100,
            # variable_noise=0.2,
            **agent_parameters,
        )
    else:
        raise ValueError(f"agent_name '{agent_name}' is not recognized")

    return agent, agent_parameters


class CartpoleRecommender:
    """

    """

    def __init__(self, cartpole_agent):
        """
        Parameters
        ----------
        cartpole_agent: Tensorforce Agent
        """
        self.cartpole_agent = cartpole_agent
        self.action = None
        self.episode_count = 0
        self.total_reward = 0.0

    def tell(self, independent_values, dependent_values):
        state = dependent_values[0]
        reward = dependent_values[1]
        terminal = dependent_values[2]
        state_after_reset = dependent_values[3]
        self.cartpole_agent.observe(reward=reward, terminal=terminal)
        if terminal > 0:
            self.total_reward = 0.0
            self.episode_count += 1
            state = state_after_reset

        self.action = self.cartpole_agent.act(states=state)

    def tell_many(self, independent_values_list, dependent_values_list):
        if len(independent_values_list) > 1:
            raise ValueError
        self.tell(
            independent_values=independent_values_list[0],
            dependent_values=dependent_values_list[0],
        )

    def ask(self, n, tell_pending=True):
        return [self.action]
