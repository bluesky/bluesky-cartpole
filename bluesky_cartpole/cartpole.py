import math

import numpy as np

from tensorforce import Environment
from tensorforce.agents import Agent

from ophyd import Component as Cpt, Device, Signal
from ophyd.status import Status


class CartPole(Device):
    """
    An ophyd Device adapting the tensorforce Environment API to the Bluesky data acquisition API.

    Instances of this class have a tensorforce cartpole Environment intended
    to be used for training an agent on the cartpole game within a Bluesky run.
    """

    # agent actions will be sent in to the
    # cartpole environment through this signal
    action = Cpt(Signal, value=0)

    # the results of the latest agent action
    # are made available by these signals
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
        """
        This method is called before starting a new training episode.
        """
        state_after_reset_ = self.cartpole_env.reset()
        self.state_after_reset.put(state_after_reset_)
        return [self]

    def trigger(self):
        """
        Perform one training step:
          - read the next agent action from the self.action signal
          - execute that action in the cartpole environment
          - record the new state of the environment
          - record the agent's reward
          - record whether the cartpole episode has terminated
          - if the game has terminated reset the cartpole environment

        Returns
        -------
        action_status: Status
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
            # the game has ended, so reset the environment
            self.state_after_reset.put(self.cartpole_env.reset())
        else:
            # the game is not over yet, so self.state_after_reset has no information
            self.state_after_reset.put(
                np.asarray([math.nan, math.nan, math.nan, math.nan])
            )

        action_status = Status()
        action_status.set_finished()
        return action_status

    def unstage(self):
        """
        There is no work to be done after training is over.
        """
        return [self]


def get_cartpole_agent(agent_name, cartpole_device):
    """
    Build a new agent for the specified cartpole device.

    It would probably make more sense to pass agent_parameters
    as a parameter to this function.

    Parameters
    ----------
    agent_name: str
        an identifier this function recognizes: "a2c" or "ppo"
    cartpole_device:

    Return
    ------
        a tensorforce Agent
    """
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
            # the cartpole environment will supply argument max_episode_timesteps
            # max_episode_timesteps=max_turns,
            **agent_parameters,
        )
    elif agent_name == "ppo":
        agent_parameters = dict(
            batch_size=10,
            variable_noise=0.1,
        )
        agent = Agent.create(
            agent="ppo",
            environment=cartpole_device.cartpole_env,
            **agent_parameters,
        )
    else:
        raise ValueError(f"agent_name '{agent_name}' is not recognized")

    return agent, agent_parameters


class CartpoleRecommender:
    """
    A bluesky-adaptive recommender.
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
