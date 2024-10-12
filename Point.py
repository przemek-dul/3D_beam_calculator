import numpy as np


class Point:
    def __init__(self, x: float, y: float, z: float, index: int = None):
        self.x = x
        self.y = y
        self.z = z
        self.index = index
        self.node_number = None  # index of the node of same coordinates as the point
        self.point_vector = np.array([x, y, z])

        self._check_input()

    def _check_input(self):

        c1 = type(self.x) == float or type(self.x) == int
        c2 = type(self.y) == float or type(self.y) == int
        c3 = type(self.z) == float or type(self.z) == int

        if not(c1 and c2 and c3):
            raise TypeError("coordinates of points must be FLOAT or INT")

        if self.index is not None:
            if type(self.index) != int:
                raise TypeError("index must be INT")
            elif self.index < 0:
                raise AttributeError("index must be equal or greater than 0")
