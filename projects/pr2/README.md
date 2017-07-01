Project 2 - RLDM - CS 7642
=============

https://github.gatech.edu/mmendiola3/rldm-project-2

GID: mmendiola3

# Requirements

- Python
- numpy
- matplotlib
- keras
- tensorflow


# Running code

Training is run with the following command. This will run until average reward reaches 200. The model will be saved after ever training run to 'model.h5'. When the agent reaches an average reward of 200, it will save the final model to 'final-model.h5'

    python q.py


The following command runs the test on the final agent model and weights. It requires 'final-model.h5'. It will run 100 episodes with rendering enabled.

    python test.py
