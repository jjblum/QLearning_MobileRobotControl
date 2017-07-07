import numpy as np
from QValueFunction import QValueFunction


class QFunctionApproximator(QValueFunction):
    def __init__(self, state_space_dims=1, action_space_dims=1):
        super(QFunctionApproximator, self).__init__(state_space_dims, action_space_dims)

    def bestAction(self, state):
        return

    def queryQ(self, state):
        return

    def reversePlayback(self, experience_queue):
        self.experience_count += len(experience_queue)
        experience_queue.reverse()
        # playback the reversed queue
        return


def main():
    print "Q"


if __name__ == "__main__":
    main()
