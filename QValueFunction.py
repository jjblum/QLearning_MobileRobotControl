import abc


class QValueFunction(object):
    # abstract class, an interface for various ways of representating the Q value function for reinforcement learning algorithms
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._experience_count = 0

    @abc.abstractmethod
    def reversePlayback(self, experience_queue):
        # apply the list of (s, a, r, s') experiences back to the Q function
        return

    @property
    def experience_count(self):
        return self._experience_count

    @experience_count.setter
    def experience_count(self, experience_cout_in):
        self._experience_count = experience_cout_in