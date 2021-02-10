================
bluesky-cartpole
================

.. image:: https://img.shields.io/travis/jklynch/bluesky-cartpole.svg
        :target: https://travis-ci.org/jklynch/bluesky-cartpole

.. image:: https://img.shields.io/pypi/v/bluesky-cartpole.svg
        :target: https://pypi.python.org/pypi/bluesky-cartpole


Train a cartpole agent with bluesky and ophyd!

* Free software: 3-clause BSD license

Install
-------

::

  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip wheel
  pip install git+https://github.com/bluesky/bluesky-adaptive.git@c3ebe70a39f6f54957d49df27654040853de5f50
  git clone git@github.com:bluesky/bluesky-cartpole.git
  cd bluesky-cartpole
  pip install -e .

Requirements
------------

bluesky-carpole requires a running MongoDB server.

Run
---

bluesky-cartpole --agent-name ppo --episode-count 100
