from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from databroker import Broker

from bluesky_cartpole.cartpole_plan import train_cartpole_agent


def run():
    RE = RunEngine()

    bec = BestEffortCallback()

    RE.subscribe(bec)

    db = Broker.named("temp")

    # Insert all metadata/data captured into db.
    RE.subscribe(db.insert)

    RE(train_cartpole_agent(episode_count=1000))


if __name__ == "__main__":
    run()
