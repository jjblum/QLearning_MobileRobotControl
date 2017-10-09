import abc  # abstract base classes
import numpy as np
import Controllers


def absoluteAngleDifference(angle1, angle2):
    while angle1 < 0.:
        angle1 += 2*np.pi
    while angle2 < 0.:
        angle2 += 2*np.pi
    angle1 = np.mod(angle1, 2*np.pi)
    angle2 = np.mod(angle2, 2*np.pi)
    return np.abs(angle1 - angle2)


class Strategy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, boat):
        self._boat = boat
        self._controller = None
        self._finished = False  # setting this to True does not necessarily mean a strategy will terminate
        self._t = boat.time
        self._strategy = self  # returns self by default (unless it is a nested strategy or sequence)
        self._strategies = list()  # not relevant for basic strategies

    @abc.abstractmethod
    def idealState(self):
        # virtual function, uses information to return an ideal state
        # this will be used for fox-rabbit style control
        return

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_in):
        self._strategy = strategy_in

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished_in):
        self._finished = finished_in

    def updateFinished(self):
        self.strategy.finished = self.strategy.controller.finished

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, controller_in):
        self._controller = controller_in

    def actuationEffortFractions(self):
        return self.controller.actuationEffortFractions()

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t
        if self._controller is not None:
            self.controller.time = t

    @property
    def boat(self):
        return self._boat

    @boat.setter
    def boat(self, boat_in):
        self._boat = boat_in


class StrategySequence(Strategy):
    """
        strategySequence: list of (class, (inputs)) stategies to be instantiated
        strategy: drills down to the lowest level current strategy
        strategies: a list of instantiated strategies

        We delay the instantiation in order to provide the most up to date system state for the later strategies.
        This is important for Executors that must make strategy choices based on system state.
        Previously, when there was just a simple list of strategies, this would instantiate all of them at once.
    """
    def __init__(self, boat, sequence):
        super(StrategySequence, self).__init__(boat)
        self._strategySequence = sequence
        self._currentStrategy = 0  # index of the current strategy
        self._strategies = list()
        self.start(self._currentStrategy)

    def start(self, currentStrategyIndex):
        # instantiate a strategy from the uninstantiated sequence
        self._strategies.append(self._strategySequence[self._currentStrategy][0](
                    *self._strategySequence[self._currentStrategy][1]))
        self._strategy = self._strategies[-1]
        self.controller = self.strategy.controller

    @property
    def strategySequence(self):
        return self._strategySequence

    @strategySequence.setter
    def strategySequence(self, strategySequence_in):
        self._strategySequence = strategySequence_in

    @property
    def strategies(self):
        return self._strategies

    # override
    def actuationEffortFractions(self):
        return self._strategies[-1].actuationEffortFractions()

    # override
    def updateFinished(self):
        """
        Switch to next strategy if the last one has been finished
        """
        self.time = self.boat.time
        self._strategies[-1].time = self.boat.time
        self._strategies[-1].updateFinished()
        if self._strategies[-1].finished and \
                self._currentStrategy < len(self.strategySequence) - 1:
            self._currentStrategy += 1
            # must manually update strategy and controller!
            self._strategies.append(self._strategySequence[self._currentStrategy][0](
                    *self._strategySequence[self._currentStrategy][1]))
            self._strategy = self._strategies[-1]
            self.controller = self.strategy.controller
        if self._strategies[-1].finished:
            # sequence is finished when last strategy in a sequence is finished
            self.finished = True

    def idealState(self):
        return self._strategies[-1].idealState()


class DoNothing(Strategy):
    # a strategy that prevents actuation
    def __init__(self, boat):
        super(DoNothing, self).__init__(boat)
        self.controller = Controllers.DoNothing()

    def idealState(self):
        return np.zeros((6,))


class DestinationOnly(Strategy):
    # a strategy that only returns the final destination location
    def __init__(self, boat, destination, positionThreshold=1.0, controller_name="PointAndShoot"):
        super(DestinationOnly, self).__init__(boat)
        self._destinationState = destination
        if controller_name == "PointAndShoot":
            THRUST_PID = [0.5, 0, 0]  #[0.5, 0.01, 10.00]  # P, I, D
            HEADING_PID = [0.7, 0, 0.9]  #[1.0, 0.0, 1.0]  # P, I, D
            HEADING_ERROR_SURGE_CUTOFF_ANGLE = 180.0  # [degrees of heading error at which thrust is forced to be zero, follows a half-cosine shape]
            self.controller = Controllers.PointAndShootPID(boat, THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, positionThreshold)
        elif controller_name == "QLearnPointAndShoot":
            self.controller = Controllers.QLearnPointAndShoot(boat)

    @property
    def destinationState(self):
        return self._destinationState  # as of now, even a high level strategy needs to have a handle to the controller it will ultimately use

    @destinationState.setter
    def destinationState(self, destinationState_in):
        if len(destinationState_in) == 6:
            self._destinationState = destinationState_in
        elif len(destinationState_in) == 3:
            # assuming they are using x, y, th
            state = np.zeros((6,))
            state[0] = destinationState_in[0]
            state[1] = destinationState_in[1]
            state[4] = destinationState_in[2]
            self._destinationState = state
        elif len(destinationState_in) == 2:
            # assuming they are using x, y
            state = np.zeros((6,))
            state[0] = destinationState_in[0]
            state[1] = destinationState_in[1]
            self._destinationState = state

    def idealState(self):
        # self.boat.plotData = np.atleast_2d(np.array([[self.boat.state[0], self.boat.state[1]], [self._destinationState[0], self._destinationState[1]]]))
        self.controller.idealState = self.destinationState  # update this here so the controller doesn't need to import Strategies


class LineFollower(Strategy):
    def __init__(self, boat, destination, positionThreshold=1.0, controller_name="PointAndShoot"):
        super(LineFollower, self).__init__(boat)
        self._x0 = boat.state[0]
        self._x1 = destination[0]
        self._y0 = boat.state[1]
        self._y1 = destination[1]
        self._dx = self._x1-self._x0
        self._dy = self._y1-self._y0
        self._th = np.arctan2(self._dy, self._dx)
        self._L = np.sqrt(np.power(self._dx, 2.)+np.power(self._dy, 2.))
        self._destination = destination
        self._lookAhead = 5.0
        self._positionThreshold = positionThreshold
        if controller_name == "PointAndShoot":
            THRUST_PID = [0.5, 0, 0]  #[0.5, 0.01, 10.00]  # P, I, D
            HEADING_PID = [0.7, 0, 0.5]  #[1.0, 0.0, 1.0]  # P, I, D
            HEADING_ERROR_SURGE_CUTOFF_ANGLE = 180.0  # [degrees of heading error at which thrust is forced to be zero, follows a half-cosine shape]
            self.controller = Controllers.PointAndShootPID(boat, THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, positionThreshold)
        elif controller_name == "QLearnPointAndShoot":
            self.controller = Controllers.QLearnPointAndShoot(boat)

    def idealState(self):
        state = np.zeros((6,))
        # project point onto line
        x = self.boat.state[0]
        dx = x - self._x0
        y = self.boat.state[1]
        dy = y - self._y0
        th = np.arctan2(dy, dx)
        dth = np.abs(self._th - th)
        currentL = np.linalg.norm(np.array([dx, dy]))*np.cos(dth)
        distance = np.linalg.norm(np.array([dx, dy]))*np.sin(dth)
        # self._lookAhead = 5.*(1.-np.tanh(0.1*distance))
        # print "distance = {:.2f} m   lookAhead = {:.2f} m".format(distance, self._lookAhead)
        projected_state = np.array([self._x0 + currentL*np.cos(self._th), self._y0 + currentL*np.sin(self._th)])
        if (currentL + self._lookAhead) > self._L:
            lookaheadState = np.array([self._x1, self._y1])
        else:
            lookaheadState = projected_state + np.array([self._lookAhead*np.cos(self._th), self._lookAhead*np.sin(self._th)])
        state[0] = lookaheadState[0]
        state[1] = lookaheadState[1]
        boatToLookahead = np.array([lookaheadState[0] - x, lookaheadState[1] - y])
        boatToLookaheadAngle = np.arctan2(boatToLookahead[1], boatToLookahead[0])
        state[4] = boatToLookaheadAngle
        self.controller.idealState = state
