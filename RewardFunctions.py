import numpy as np

"""
Various reward functinos associated with mobile robot navigation
"""


def reward_reachGoalSparse(state, action, destination, position_threshold=0.0, reward_value=1.0, velocity_threshold=np.inf):
    """
    Receive a nonzero reward only upon reaching the destination. This is a sparse reward.
    :param state:
    :param action:
    :param destination:
    :param position_threshold: the distance around the destination where a nonzero reward exists
    :param reward_value: the maximum reward value that can be achieved (accounting for nontrivial uses of position and velocity thresholds)
    :param velocity_threshold: the velocity below which a nonzero reward exists. Used so that a vehicle tries to arrive at a point, not rush through it.
    :return: a scalar representing the reward the learner should receive for the queried state and action
    """
    return 0


def reward_reachGoalDense(state, action, destination, position_length_scale=1.0, reward_value=1.0, velocity_length_scale=np.inf):
    """
    Receive a nonzero reward as a function of distance to the destination. This is a dense reward.
    :param state:
    :param action:
    :param destination:
    :param position_length_scale: used to shape the reward surface as a function of position
    :param reward_value: the maximum value that can be achieved
    :param velocity_length_scale: used to shape the reward surface as a function of velocity
    :return:
    """
    return 0
