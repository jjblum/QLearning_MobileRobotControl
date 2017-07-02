import numpy as np
import abc
import math


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def dragDown(boat):
    return boat.design.interpolateDragDown(boat.state[2])


class UniversalPID(object):
    def __init__(self, boat, P, I, D, t, name):
        self._boat = boat
        self._P = P
        self._I = I
        self._D = D
        self._t = t
        self._tOld = t
        self._errorDerivative = 0.0
        self._errorAccumulation = 0.0
        self._errorOld = 0.0
        self._name = name

    def signal(self, error, t):
        dt = t - self._t
        self._t = t
        self._errorDerivative = 0.0
        if dt > 0:
            self._errorDerivative = (error - self._errorOld)/dt
        self._errorAccumulation += dt*error
        #if self._name == "heading_PID":
            #print "{}: e = {}, P term = {}, I term = {}, D term = {}".format(self._name, error, self._P*error, self._I*self._errorAccumulation, self._D*self._errorDerivative)
        return self._P*error + self._I*self._errorAccumulation + self._D*self._errorDerivative


class Controller(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._t = 0.0
        self._boat = None
        self._idealState = []
        self._thrustFraction = 0.0
        self._momentFraction = 0.0
        self._finished = False

    @abc.abstractmethod
    def actuationEffortFractions(self):
        # virtual function, uses current state and ideal state to generate actuation effort
        # PID control, trajectory following, etc.
        return

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t

    @property
    def boat(self):
        return self._boat

    @boat.setter
    def boat(self, boat_in):
        self._boat = boat_in

    @property
    def idealState(self):
        return self._idealState

    @idealState.setter
    def idealState(self, idealState_in):
        self._idealState = idealState_in

    @property
    def thrustFraction(self):
        return self._thrustFraction

    @thrustFraction.setter
    def thrustFraction(self, thrustFraction_in):
        self._thrustFraction = thrustFraction_in

    @property
    def momentFraction(self):
        return self._momentFraction

    @momentFraction.setter
    def momentFraction(self, momentFraction_in):
        self._momentFraction = momentFraction_in

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished_in):
        self._finished = finished_in


class DoNothing(Controller):
    def __init__(self):
        super(DoNothing, self).__init__()

    def actuationEffortFractions(self):
        return 0.0, 0.0


class PointAndShootPID(Controller):

    def __init__(self, boat, thrust_PID, heading_PID, headingErrorSurgeCutoff, positionThreshold_in=1.0):
        super(PointAndShootPID, self).__init__()
        self._boat = boat
        self.time = boat.time
        self._positionThreshold = positionThreshold_in
        self._positionPID = UniversalPID(boat, thrust_PID[0], thrust_PID[1], thrust_PID[2], boat.time, "position_PID")
        self._headingPID = UniversalPID(boat, heading_PID[0], heading_PID[1], heading_PID[2], boat.time, "heading_PID")
        self._headingErrorSurgeCutoff = headingErrorSurgeCutoff*math.pi/180.0  # thrust signal rolls off as a cosine, hitting zero here

    def positionThreshold(self):
        return self._positionThreshold

    def positionThreshold(self, positionThreshold_in):
        self._positionThreshold = positionThreshold_in

    def actuationEffortFractions(self):
        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state

        error_x = self.idealState[0] - state[0]
        error_y = self.idealState[1] - state[1]
        error_pos = math.sqrt(math.pow(error_x, 2.0) + math.pow(error_y, 2.0))

        # if the position error is less than some threshold and velocity is near zero, turn thrustFraction to 0
        if error_pos < self._positionThreshold:
            # because this is where we might set finished to True, it
            # needs to be before any other returns that might make it impossible to reach
            self.finished = True
            return 0.0, 0.0

        if self.finished:
            return 0.0, 0.0

        angleToGoal = math.atan2(error_y, error_x)
        error_th = wrapToPi(state[4] - angleToGoal)  # error between heading and heading to idealState

        error_th_signal = self._headingPID.signal(error_th, self.boat.time)
        error_pos_signal = self._positionPID.signal(error_pos, self.boat.time)

        self.time = self.boat.time

        clippedAngleError = np.clip(math.fabs(error_th), 0.0, self._headingErrorSurgeCutoff)
        thrustReductionRatio = math.cos(math.pi/2.0*clippedAngleError/self._headingErrorSurgeCutoff)
        momentFraction = np.clip(error_th_signal, -1.0, 1.0)
        #thrustFraction = np.clip(error_pos_signal, -thrustReductionRatio, thrustReductionRatio)
        thrustFraction = np.clip(error_pos_signal, -1.0, 1.0)
        thrustFraction = thrustFraction*thrustReductionRatio

        return thrustFraction, momentFraction


class QLearnPointAndShoot(Controller):
    def __init__(self, boat, positionThreshold_in=1.0, learning_rate=0.2):
        super(QLearnPointAndShoot, self).__init__()
        self._boat = boat
        self.time = boat.time
        self._positionThreshold = positionThreshold_in
        self._Q = boat.Q  # the model representation of the value function

    def actuationEffortFractions(self):
        # use state of the boat and the ideal state to determine an action to take
        # TODO

        state = self.boat.state
        error_x = self.idealState[0] - state[0]
        error_y = self.idealState[1] - state[1]
        error_pos = math.sqrt(math.pow(error_x, 2.0) + math.pow(error_y, 2.0))

        # if the position error is less than some threshold and velocity is near zero, turn thrustFraction to 0
        if error_pos < self._positionThreshold:
            self.finished = True
            return 0.0, 0.0

        return 0., 0.
