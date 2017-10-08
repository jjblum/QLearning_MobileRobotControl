import numpy as np
from QValueFunction import QValueFunction
# k-d tree? Idk, it is already 10 dimensional. --> sklearn.neighbors.KNeighborsRegressor
# hat matrix (see bottom right of page 2 of Practical Reinforcement Learning in Continuous Spaces)

# KISS-GP or (MSGP, the extension) lets you create GPs when you have many dimensions
# But wouldn't I need a massive amount of points in the fixed grid if I really have 10 dimensions?
# At that point I should probably just use a NN due to its simplicity, as long as it could be trained
#

"""
steps to providing Q value for performing action a while in state s
0) default "I don't know" value for Q. A high value would encourage random action, which may or may not be suitable for a mobile robot.
   A low value would cause the robot to stick more closely to sequences of actions that reached a nonzero reward.
1) playback experiences, creating estimated Q values as a function of a concatenated state and action vector
   Every time you use a single experience, it will need to query Q(s', a*) to get the estimated optimal Q value at next state s'.

2)
"""


# scaling: the most common scaling technique is subtracting the mean and dividing by the standard deviation for each dimension


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
