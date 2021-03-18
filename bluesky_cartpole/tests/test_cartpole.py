import pprint

from bluesky.tests.utils import DocCollector as DocumentCollector

from bluesky_adaptive.per_event import (
    recommender_factory,
    adaptive_plan,
)

from bluesky_cartpole.cartpole import CartPole, CartpoleRecommender, get_cartpole_agent


def test_cartpole_device():
    cartpole = CartPole()

    description = cartpole.describe()
    pprint.pprint(description)

    cartpole.stage()

    cartpole.trigger()
    pprint.pprint(cartpole)

    cartpole_state = cartpole.read()
    pprint.pprint(cartpole_state)


def test_per_event_adaptive_plan(RE):

    cartpole_device = CartPole(max_episode_timesteps=10)
    cartpole_agent, agent_parameters = get_cartpole_agent(
        agent_name="a2c", cartpole_device=cartpole_device
    )
    cartpole_recommender = CartpoleRecommender(cartpole_agent=cartpole_agent)
    to_recommender, from_recommender = recommender_factory(
        cartpole_recommender,
        independent_keys=[cartpole_device.action.name],
        # recommender expects the dependent values in this order
        dependent_keys=[
            cartpole_device.next_state.name,
            cartpole_device.reward.name,
            cartpole_device.terminal.name,
            cartpole_device.state_after_reset.name,
        ],
    )

    # stage the cartpole device to reset the underlying cartpole environment
    cartpole_device.stage()
    action = cartpole_agent.act(states=cartpole_device.state_after_reset.get())

    RE(
        adaptive_plan(
            dets=[cartpole_device],
            first_point={cartpole_device.action: action},
            to_recommender=to_recommender,
            from_recommender=from_recommender,
        )
    )

    # pprint.pprint(f"cartpole_agent.get_variables(): {cartpole_agent.get_variables()}")


def __test_cartpole_recommender(RE, hw):

    recommender = CartpoleRecommender()

    cb, queue = recommender_factory(recommender, ["motor"], ["det"])
    dc = DocumentCollector()

    RE.subscribe(dc.insert)
    RE(
        adaptive_plan(
            [hw.det], {hw.motor: 0}, to_recommender=cb, from_recommender=queue
        )
    )

    assert len(dc.start) == 1
    assert len(dc.event) == 1
    (events,) = dc.event.values()
    assert len(events) == 4
