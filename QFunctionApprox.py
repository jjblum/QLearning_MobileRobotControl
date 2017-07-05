import numpy as np
from QValueFunction import QValueFunction


class QFunctionApproximator(QValueFunction):
    def __init__(self):
        super(QFunctionApproximator, self).__init__()

    def reversePlayback(self, experience_queue):
        self._experience_count += len(experience_queue)
        return


def main():
    print "Q"


if __name__ == "__main__":
    main()
