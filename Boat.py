import numpy as np
import math
import copy
import Strategies
import Designs
import QFunctionApprox
import RewardFunctions

__author__ = 'jjb'

# the model representation of the value function, shared by both the example PID boat and the Q learning boat
_Q_ = QFunctionApprox.QFunctionApproximator(state_space_dims=8, action_space_dims=2)


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

    def __init__(self, t=0.0, name="boat", design=Designs.TankDriveDesign()):
        self._t = t  # current time [s]
        self._name = name
        self._state = np.zeros((6,))
        self._sourceLocation = np.zeros((2,))  # where the boat started from
        self._destinationLocation = np.zeros((2,))  # where the boat is going (typically the next waypoint)
        # state: [x y u w th thdot]
        self._thrustSurge = 0.0  # surge thrust [N]
        self._thrustSway = 0.0  # sway thrust (zero for tank drive) [N]
        self._moment = 0.0  # [Nm]
        self._thrustFraction = 0.0
        self._momentFraction = 0.0
        self._strategy = Strategies.DoNothing(self)
        self._design = design
        self._plotData = None  # [x, y] data used to display current actions
        self._controlHz = 5  # the number of times per second the boat is allowed to change its signal, check if strategy is finished, and create Q experiences
        self._lastControlTime = 0
        self._Q = _Q_
        self._Qstate = np.zeros((8,))  # [u w alpha delta phi alphadot deltadot phidot]
        self._QlastState = np.zeros((8,))  # the state "s" in the experience (s, a, r, s')
        self._QlastAction = np.zeros((2,))  # the action "a" in the experience (s, a, r, s'), [m0_signal m1_signal]
        # alpha = progress along line between origin and waypoint, normalized by the length of the line
        # delta = distance to the line (projection of boat onto the line)
        # phi = angle of the line with respect to the surge direction in the body frame
        # mX_signal = raw signal value of actuator X
        self._QExperienceQueue = list()  # a list containing the current set of experiences in the order they were created
        self._lastExperienceTime = 0

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t
        self.strategy.time = t

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name_in):
        self._name = name_in

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state_in):
        self._state = state_in

    @property
    def sourceLocation(self):
        return self._sourceLocation

    @sourceLocation.setter
    def sourceLocation(self, sourceLocation_in):
        self._sourceLocation = sourceLocation_in

    @property
    def destinationLocation(self):
        return self._destinationLocation

    @destinationLocation.setter
    def destinationLocation(self, destinationLocation_in):
        self._destinationLocation = destinationLocation_in

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
            # print self._name + ": control() iteration, t = {}".format(self._t)
            # print "Boat control triggered, t = {:.2f}".format(self.time)
            self.strategy.updateFinished()

            self._QlastAction = self.design.actuatorSignals
            self.createExperience()  # run this before changing control

            self.strategy.idealState()
            self._thrustFraction, self._momentFraction = self.strategy.actuationEffortFractions()
            self._lastControlTime = self.time

            # TODO: create an exponential delay so that changing signals does create instant changes in thrust and moment
            self.thrustSurge, self.thrustSway, self.moment = \
                self.design.thrustAndMomentFromFractions(self._thrustFraction, self._momentFraction)

    def sourceToDestinationLine(self):
        """
        Calculate some useful information about the line between boat source and destination locations
        :return L (the length of the line
        :return theta (the angle of the line
        :return phi (the angle of the line with respect to current boat surge direction)
        :return alpha (the current progress of the boat along the line)
        """
        source_to_dest = self._destinationLocation - self._sourceLocation
        source_to_boat = self._state[0:2] - self._sourceLocation
        boat_to_dest = self._destinationLocation - self._state[0:2]
        source_to_boat_angle = np.arctan2(source_to_boat[1], source_to_boat[0])
        source_to_dest_angle = np.arctan2(source_to_dest[1], source_to_dest[0])
        dth = np.abs(source_to_dest_angle - source_to_boat_angle)  # need to use difference in angles of lines from source to dest and source to boat
        source_to_dest_length = np.linalg.norm(source_to_dest)
        source_to_boat_length = np.linalg.norm(source_to_boat)
        phi = wrapToPi(source_to_dest_angle - self._state[4])
        alpha = source_to_boat_length*np.cos(dth) / source_to_dest_length
        delta = source_to_dest_length*alpha*np.sin(dth)
        #if self._name == "pid boat":
        #    print source_to_dest_length, source_to_dest_angle, phi, alpha, delta
        return source_to_dest_length, source_to_dest_angle, phi, alpha, delta

    def distanceFromDestination(self):
        return np.linalg.norm(self._destinationLocation - self._state[0:2])

    def projectVelocityOntoSourceDestLine(self, L, phi):
        # normalize the parallel component by length of the line to create alphadot
        u = self._state[2]
        w = self._state[3]
        parallel = u*np.cos(phi) + w*np.sin(phi)
        perpendicular = -u*np.sin(phi) + w*np.cos(phi)
        return parallel/L, perpendicular

    def calculateQState(self):
        # [u w alpha delta phi alphadot deltadot phidot]  TODO: I don't think alpha and alphadot are a good idea. The length of the line is too variable.
        # [u w alpha*L delta phi alphadot*L deltadot phidot]  # multiply L back in
        L, theta, phi, alpha, delta = self.sourceToDestinationLine()
        u = self._state[2]
        w = self._state[3]
        alphadot, deltadot = self.projectVelocityOntoSourceDestLine(L, phi)
        phidot = self._state[5]  # boat's heading rate of change is the same as phidot
        self._Qstate = np.array([u, w, (1-alpha)*L, delta, phi, alphadot*L, deltadot, phidot])
        return

    def createExperience(self):
        # take previous state (s), previous action (a), current reward (r), and current state (s')
        previous_state = self._QlastState  # s
        previous_action = self._QlastAction  # a

        # calculate new Q state, s'
        self.calculateQState()
        current_state = self._Qstate
        reward = self._Q.reward(RewardFunctions.reward_reachGoalSparse, (self.distanceFromDestination(), self._state[2], 1.0, 1.0, np.inf))

        experience = (previous_state, previous_action, reward, current_state)
        """
        if self._name == "pid boat" and reward > 0:
            print self._destinationLocation
            print self._state
            print self.distanceFromDestination()
            print current_state
        """
        self._QExperienceQueue.append(experience)
        self._QlastState = self._Qstate


