import time

from tensorforce import Environment
from tensorforce.agents import Agent

from ophyd import Component as Cpt, Device, Signal
from ophyd.status import Status


class CartPole(Device):
    # read the next action from this Signal
    # there is an agent on the other side of it
    next_action = Cpt(Signal)

    def __init__(self, name="cartpole", prefix="CARTPOLE"):
        super().__init__(name=name, prefix=prefix)

        self.cartpole_env = Environment.create(environment='gym', level='CartPole-v1')
        self.next_state = None
        self.terminal = None
        self.reward = None
        self.state_after_reset = None

    def stage(self):
        return [self]

    def trigger(self):
        self.next_state, self.terminal, self.reward = self.cartpole_env.execute(
            actions=self.next_action.get()
        )
        if self.terminal == 1:
            print("terminal!")
            self.state_after_reset = self.cartpole_env.reset()
        else:
            self.state_after_reset = None

        status = Status()
        status.set_finished()
        return status

    def describe(self):
        return {
            self.name: {
                "dtype": "number",
                "shape": [],
                "source": "where it came from (PV)",
            }
        }

    def read(self):
        return {
            self.name: {
                "value": (
                    self.next_state,
                    self.terminal,
                    self.reward,
                    self.state_after_reset,
                ),
                "timestamp": time.time(),
            }
        }

    def unstage(self):
        return [self]


def get_cartpole_agent(cartpole):
    agent = Agent.create(
        agent="a2c",
        batch_size=100,  # this seems to help a2c
        exploration=0.01,  # tried without this at first
        variable_noise=0.05,
        # variable_noise=0.01 bad?
        l2_regularization=0.1,
        entropy_regularization=0.2,
        horizon=10,  # does this help a2c? yes
        environment=cartpole.cartpole_env,

        # the gym cartpole environment will supply max_episode_timesteps
        #max_episode_timesteps=max_turns,

        summarizer=dict(
            directory="data/summaries",
            # list of labels, or 'all'
            labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
            frequency=10,  # store values every 10 timesteps
        ),
    )
    return agent


