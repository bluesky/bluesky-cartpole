================
bluesky-cartpole
================

.. image:: https://github.com/bluesky/bluesky-cartpole/actions/workflows/testing.yml/badge.svg

Train a cartpole agent with bluesky and ophyd!

* Free software: 3-clause BSD license

Install
-------

::

  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip wheel
  git clone git@github.com:bluesky/bluesky-cartpole.git
  cd bluesky-cartpole
  pip install -e .

Requirements
------------

bluesky-carpole requires a running MongoDB server.

Run
---

bluesky-cartpole --agent-name ppo --episode-count 100
