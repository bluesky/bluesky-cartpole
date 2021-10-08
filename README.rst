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

bluesky-carpole requires a running MongoDB server.  By default it is assumed that MongoDB is 
running on `localhost` (i.e. 127.0.0.1) and the default port of 27017.

Run
---

bluesky-cartpole --agent-name ppo --episode-count 100
