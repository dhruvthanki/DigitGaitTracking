#@author: Abhijeet Kulkarni
#@breif: Say goodbye to wrapping problems!!!
import numpy as np

class Angle:
    def __init__(self, theta=0.0):
        self.a = 1.0 # real part
        self.b = 0.0 # imaginary part
        self._from_theta_to_complex(theta)

    def _from_theta_to_complex(self, theta):
        self.a = np.cos(theta)
        self.b = np.sin(theta)
    
    def _normalize(self):
        norm = np.sqrt(self.a**2 + self.b**2)
        self.a = self.a/norm
        self.b = self.b/norm

    @property
    def toRadian(self):
        return np.arctan2(self.b, self.a)
    
    @property
    def toDegree(self):
        return np.rad2deg(self.toRadian)
    
    def __add__(self, other):
        if isinstance(other, Angle):
            return Angle(self.toRadian + other.toRadian)
        elif isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian + other)
        else:
            raise TypeError("unsupported operand type(s) for +: 'Angle' and '{}'".format(type(other)))
    
    def __sub__(self, other):
        if isinstance(other, Angle):
            return Angle(self.toRadian - other.toRadian)
        elif isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian - other)
        else:
            raise TypeError("unsupported operand type(s) for -: 'Angle' and '{}'".format(type(other)))
    
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian * other)
        else:
            raise TypeError("unsupported operand type(s) for *: 'Angle' and '{}'".format(type(other)))
    
    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian / other)
        else:
            raise TypeError("unsupported operand type(s) for /: 'Angle' and '{}'".format(type(other)))
    
    def __repr__(self):
        return "Angle({})".format(self.toRadian)

    def __str__(self):
        return "Angle({})".format(self.toRadian)
