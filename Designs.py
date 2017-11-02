import numpy as np
import abc


def scale_down(m0raw, m1raw):
    # scale down such that the ratio of m0/m1 is maintained but the max absolute value is 1.0
    m0, m1 = m0raw, m1raw
    if np.abs(m0raw) > 1. or np.abs(m1raw) > 1.:
        if np.abs(m0raw) > np.abs(m1raw):
            m0 = m0raw/np.abs(m0raw)
            m1 = m1raw/np.abs(m0raw)
        else:
            m0 = m0raw/np.abs(m1raw)
            m1 = m1raw/np.abs(m1raw)
    return m0, m1


class Design(object):
    # abstract class, a design dictates how actuation fractions are translated into actual thrust and moment
    # e.g. a tank-drive propeller boat will behave differently than a vectored-thrust boat
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._mass = 0.0  # [kg]
        self._momentOfInertia = 0.0  # [kg/m^2]
        self._dragAreas = [0.0, 0.0, 0.0]  # surge, sway, rotation [m^2]
        self._dragCoeffs = [0.0, 0.0, 0.0]  # surge, sway, rotation [-]
        self._actuator_signals = np.zeros((2,))  # signal 0, signal 1

    @abc.abstractmethod
    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        # virtual, calculate the thrust and moment a specific boat design can accomplish
        return

    @property
    def maxForwardThrust(self):
        return self._maxForwardThrust

    @property
    def mass(self):
        return self._mass

    @property
    def momentOfInertia(self):
        return self._momentOfInertia

    @property
    def dragAreas(self):
        return self._dragAreas

    @property
    def dragCoeffs(self):
        return self._dragCoeffs

    @property
    def dragAreas(self):
        return self._dragAreas

    @property
    def dragCoeffs(self):
        return self._dragCoeffs

    @property
    def actuatorSignals(self):
        return self._actuator_signals

    @actuatorSignals.setter
    def actuatorSignals(self, actuator_signals_in):
        self._actuator_signals = actuator_signals_in


class Lutra(Design):
    def __init__(self):
        super(Lutra, self).__init__()
        self._mass = 5.7833  # [kg]
        self._momentOfInertia = 0.7  # [kg/m^2]
        self._dragAreas = [0.0108589939, 0.0424551192, 0.0424551192]  # surge, sway, rotation [m^2]
        # self._dragCoeffs = [0.258717640651218, 1.088145891415693, 0.048292066650533]  # surge, sway, rotation [-]
        self._dragCoeffs = [0.258717640651218, 1.088145891415693, 0.2]  # surge, sway, rotation [-]
        #  self._dragCoeffs = [1.5, 1.088145891415693, 2.0]  # surge, sway, rotation [-]
        #self._dragCoeffs = [0.258717640651218, 1.088145891415693, 2.0]  # surge, sway, rotation [-]


class AirboatDesign(Lutra):
    def __init__(self):
        super(AirboatDesign, self).__init__()
        self._maxForwardThrust = 20.0  # [N]
        self._maxBackwardThrust = 7.0  # [N]
        self._momentArm = 0.5 # [m]

    def thrustAndMomentFromSignals(self):
        m0 = self._actuator_signals[0]
        s0 = -self._actuator_signals[1]
        if np.abs(m0) > 1:
            m0 = 1.*np.sign(m0)
        if m0 > 0:
            thrust = m0*self._maxForwardThrust
        else:
            thrust = m0*self._maxBackwardThrust
        if np.abs(s0) > 1:
            s0 = 1.*np.sign(s0)
        angle = s0*75.*np.pi/180.
        #print "Airfan signal = {:f},  angle = {:f} deg".format(s0, angle*180./np.pi)
        return thrust*np.cos(angle), thrust*np.sin(angle), thrust*np.sin(angle)*self._momentArm

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        self._actuator_signals = thrustFraction, momentFraction
        return self.thrustAndMomentFromSignals()


class TankDriveDesign(Lutra):
    def __init__(self):
        super(TankDriveDesign, self).__init__()
        self._maxForwardThrustPerMotor = 25.0  # [N]
        self._maxBackwardThrustPerMotor = 10.0  # back-driving motors is much weaker, or a propguard lowers this thrust
        self._momentArm = 0.3556  # distance between the motors [m]
        # below 1 m/s, you should probably just turn in place!

    def thrustAndMomentFromSignals(self):
        m0 = self._actuator_signals[0]
        m1 = self._actuator_signals[1]
        if m0 > 0:
            #t0 = 0.75*self._maxForwardThrustPerMotor*m0  # imbalanced, as if the prop has chipped
            t0 = self._maxForwardThrustPerMotor*m0
        else:
            t0 = self._maxBackwardThrustPerMotor*m0

        if m1 > 0:
            t1 = self._maxForwardThrustPerMotor*m1
        else:
            t1 = self._maxBackwardThrustPerMotor*m1
        thrustSurge = t0 + t1
        moment = (t1 - t0)/2.0*self._momentArm
        return thrustSurge, 0.0, moment

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        self.actuatorSignals = scale_down(thrustFraction + momentFraction, thrustFraction - momentFraction)
        return self.thrustAndMomentFromSignals()
