import numpy as np
import math
import copy
import Strategies
import Designs

__author__ = 'jjb'


_Q_ = None  # the model representation of the value function


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrapTo2Pi(angle):
    angle = wrapToPi(angle)
    if angle < 0:
        angle += 2*np.pi
    return angle



def ode(state, t, boat):
    # derivative of state at input state and time
    # this is in Boat, not Design, because only the forces and moment are relevant
    rho = 1000.0  # density of water [kg/m^3]
    u = state[2]
    w = state[3]
    th = state[4]
    thdot = state[5]
    au = boat.design.dragAreas[0]
    aw = boat.design.dragAreas[1]
    ath = boat.design.dragAreas[2]
    cu = boat.design.dragCoeffs[0]
    cw = boat.design.dragCoeffs[1]
    cth = boat.design.dragCoeffs[2]
    qdot = np.zeros((6,))
    qdot[0] = u*math.cos(th) - w*math.sin(th)
    qdot[1] = u*math.sin(th) + w*math.cos(th)
    qdot[2] = 1.0/boat.design.mass*(boat.thrustSurge - 0.5*rho*au*cu*math.fabs(u)*u)
    qdot[3] = 1.0/boat.design.mass*(boat.thrustSway - 0.5*rho*aw*cw*math.fabs(w)*w)
    qdot[4] = thdot
    qdot[5] = 1.0/boat.design.momentOfInertia*(boat.moment - 0.5*rho*ath*cth*math.fabs(thdot)*thdot)

    # linear friction, only dominates when boat is moving slowly
    #if u < 0.25:
    #    qdot[2] -= 1.0/boat.design.mass*5.0*u - np.sign(u)*0.001
    #if w < 0.25:
    #    qdot[3] -= 1.0/boat.design.mass*5.0*w - np.sign(w)*0.001
    #if thdot < math.pi/20.0:  # ten degrees per second
    #    qdot[5] -= 1.0/boat.design.momentOfInertia*5.0*thdot - np.sign(thdot)*0.001

    return qdot


class Boat(object):

    def __init__(self, t=0.0):
        self._t = t  # current time [s]
        self._state = np.zeros((6,))
        self._idealState = np.zeros((6,))  # a "rabbit" boat to chase
        # state: [x y u w th thdot]
        self._thrustSurge = 0.0  # surge thrust [N]
        self._thrustSway = 0.0  # sway thrust (zero for tank drive) [N]
        self._moment = 0.0  # [Nm]
        self._thrustFraction = 0.0
        self._momentFraction = 0.0
        self._strategy = Strategies.DoNothing(self)
        self._design = Designs.TankDriveDesign()
        self._plotData = None  # [x, y] data used to display current actions
        self._Q = _Q_
        self._controlHz = 5  # the number of times per second the boat is allowed to change its signal
        self._lastControlTime = 0

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t
        self.strategy.time = t

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state_in):
        self._state = state_in

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type_in):
        self._type = type_in

    @property
    def thrustSurge(self):
        return self._thrustSurge

    @thrustSurge.setter
    def thrustSurge(self, thrustSurge_in):
        self._thrustSurge = thrustSurge_in

    @property
    def thrustSway(self):
        return self._thrustSway

    @thrustSway.setter
    def thrustSway(self, thrustSway_in):
        self._thrustSway = thrustSway_in

    @property
    def moment(self):
        return self._moment

    @moment.setter
    def moment(self, moment_in):
        self._moment = moment_in

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_in):
        self._strategy = strategy_in

    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, design_in):
        self._design = design_in

    @property
    def plotData(self):
        return self._plotData

    @plotData.setter
    def plotData(self, plotData_in):
        self._plotData = plotData_in

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q_in):
        self._Q = Q_in

    def __str__(self):
        return "Boat {ID}: {T} at X = {X}, Y = {Y}, TH = {TH}".format(ID=self.uniqueID,
                                                                      X=self.state[0][0],
                                                                      Y=self.state[1][0],
                                                                      T=self.type,
                                                                      TH=self.state[4][0])

    def distanceToPoint(self, point):
        return np.sqrt(np.power(self._state[0] - point[0], 2) + np.power(self._state[1] - point[1], 2))

    def globalAngleToPoint(self, point):
        """
        Angle to a point with respect to the global x direction
        """
        dx = point[0] - self._state[0]
        dy = point[1] - self._state[1]
        return np.arctan2(dy, dx)

    def localAngeToPoint(self, point):
        """
        Angle to a point with respect to the surge direction
        """
        ga = self.globalAngleToPoint(point)
        if ga < 0:
            ga += 2*np.pi
        a = copy.deepcopy(self._state[4])
        if a < 0:
            a += 2*np.pi
        return wrapToPi(ga - a)

    def control(self):
        if self.time > self._lastControlTime + 1./self._controlHz:
            # print "Boat control triggered, t = {:.2f}".format(self.time)
            self.strategy.updateFinished()
            self.strategy.idealState()
            self._thrustFraction, self._momentFraction = self.strategy.actuationEffortFractions()
            self._lastControlTime = self.time
        self.thrustSurge, self.thrustSway, self.moment = \
            self.design.thrustAndMomentFromFractions(self._thrustFraction, self._momentFraction)
