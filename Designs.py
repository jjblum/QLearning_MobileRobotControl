import numpy as np
import abc


class Design(object):
    # abstract class, a design dictates how actuation fractions are translated into actual thrust and moment
    # e.g. a tank-drive propeller boat will behave differently than a vectored-thrust boat
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._mass = 0.0  # [kg]
        self._momentOfInertia = 0.0  # [kg/m^2]
        self._dragAreas = [0.0, 0.0, 0.0]  # surge, sway, rotation [m^2]
        self._dragCoeffs = [0.0, 0.0, 0.0]  # surge, sway, rotation [-]
        self._maxHeadingRate = 0.0  # maximum turning speed [rad/s]
        self._maxForwardThrust = 0.0
        self._speedVsMinRadius = np.zeros((1, 2))  # 2 column array, speed vs. min turning radius

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


class Lutra(Design):
    def __init__(self):
        super(Lutra, self).__init__()
        self._mass = 5.7833  # [kg]
        self._momentOfInertia = 0.6  # [kg/m^2]
        self._dragAreas = [0.0108589939, 0.0424551192, 0.0424551192]  # surge, sway, rotation [m^2]
        # self._dragCoeffs = [0.258717640651218, 1.088145891415693, 0.048292066650533]  # surge, sway, rotation [-]
        self._dragCoeffs = [1.5, 1.088145891415693, 2.0]  # surge, sway, rotation [-]


class TankDriveDesign(Lutra):
    def __init__(self):
        super(TankDriveDesign, self).__init__()
        self._maxForwardThrustPerMotor = 25.0  # [N]
        self._maxBackwardThrustPerMotor = 10.0  # back-driving motors is much weaker, or a propguard lowers this thrust
        self._momentArm = 0.3556  # distance between the motors [m]
        # below 1 m/s, you should probably just turn in place!
        self._maxHeadingRate = 0.403  # [rad/s]
        self._thCoeff = 2.54832785865
        self._rCoeff = 0.401354269952
        self._u0Coeff = 0.0914788305811

    @property
    def thCoeff(self):
        return self._thCoeff

    @property
    def rCoeff(self):
        return self._rCoeff

    @property
    def u0Coeff(self):
        return self._u0Coeff

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        thrustSway = 0.0

        m0 = np.clip(thrustFraction + momentFraction, -1.0, 1.0)
        m1 = np.clip(thrustFraction - momentFraction, -1.0, 1.0)

        if m0 > 0:
            t0 = self._maxForwardThrustPerMotor*m0
        else:
            t0 = self._maxBackwardThrustPerMotor*m0

        if m1 > 0:
            t1 = self._maxForwardThrustPerMotor*m1
        else:
            t1 = self._maxBackwardThrustPerMotor*m1
        thrustSurge = t0 + t1
        moment = (t1 - t0)/2.0*self._momentArm

        return thrustSurge, thrustSway, moment