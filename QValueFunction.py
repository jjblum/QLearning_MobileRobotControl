import abc


class QValueFunction(object):
    # abstract class, an interface for various ways of representating the Q value function for reinforcement learning algorithms
    __metaclass__ = abc.ABCMeta

    def __init__(self, state_space_dims=1, action_space_dims=1):
        self._experience_count = 0
        self._Q = None
        self._state_space_dims = state_space_dims
        self._action_space_dims = action_space_dims

    @staticmethod
    def reward(rewardFunction, arguments):
        return rewardFunction(*arguments)

    @abc.abstractmethod
    def bestAction(self, state):
        # virtual, find the best action to take at state
        return

    @abc.abstractmethod
    def queryQ(self, state):
        # virtual, find the best action to take at the state "a*" and query the Q value, Q(s, a*)
        return

    @abc.abstractmethod
    def reversePlayback(self, experience_queue):
        # virtual, apply the list of (s, a, r, s') experiences back to the Q function
        return

    @property
    def experience_count(self):
        return self._experience_count

    @experience_count.setter
    def experience_count(self, experience_cout_in):
        self._experience_count = experience_cout_in

    @property
    def Q(self):
        return self._Q