import argparse

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from databroker import Broker

from bluesky_cartpole.cartpole_plan import train_cartpole_agent


def run():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--agent-name", required=True, type=str)
    arg_parser.add_argument("--episode-count", required=True, type=int)

    args = arg_parser.parse_args()

    RE = RunEngine()

    bec = BestEffortCallback()

    RE.subscribe(bec)

    db = Broker.named("bluesky-cartpole")

    # insert bluesky documents into databroker
    RE.subscribe(db.insert)

    RE(
        train_cartpole_agent(
            agent_name=args.agent_name, episode_count=args.episode_count
        )
    )


if __name__ == "__main__":
    run()
