import numpy as np
from Point import Point
from Material import Material
from Section import Section
import vg


class Line:
    def __init__(self, point1: Point, point2: Point, material: Material, section: Section,
                 direction_vector: list = None, extra_point: list = None):
        self.point1 = point1
        self.point2 = point2
        self.material = material
        self.section = section
        self.elements_index = []  # the set of indexes of elements on a line
        self.len = 0
        self.extra_point = extra_point  # point used to calculate direction vector of local z axis
        self.direction_vector = direction_vector  # direction vector of local z axis

        self.check_input()
        self.get_direction_vector()

    def get_direction_vector(self):
        """
        Returns direction vector of local z axis. If user did not define direction_vector and extra_point, default
        direction vector is [0, 0, -1].
        If extra_point is defined, approximate direction vector is calculate as the vector from extra_point and first
        point of line.
        """
        t1 = self.point2.x - self.point1.x
        t2 = self.point2.y - self.point1.z
        t3 = self.point2.z - self.point1.z

        t_vector = np.array([t1, t2, t3])

        if self.direction_vector is None and self.extra_point is None:
            self.direction_vector = np.array([0, 0, -1])
        elif self.extra_point is not None and self.direction_vector is None:

            v1 = self.extra_point[0] - self.point1.x
            v2 = self.extra_point[1] - self.point1.y
            v3 = self.extra_point[2] - self.point1.z

            v_vector = np.array([v1, v2, v3])

            n2_vector = np.cross(t_vector, v_vector)
            n1_vector = np.cross(n2_vector, t_vector)

            self.direction_vector = n1_vector

        if vg.almost_collinear(t_vector, self.direction_vector):
            pass
            #raise AttributeError("Direction vector is parallel to line")

    def check_input(self):
        if type(self.point1) != Point or type(self.point2) != Point:
            raise TypeError("argument - point must be Point")

        self.len = np.sqrt(pow(self.point1.x - self.point2.x, 2) + pow(self.point1.y - self.point2.y, 2) + pow(
            self.point1.z - self.point2.z, 2))

        if self.len == 0:
            raise ValueError("length of line is 0 - check input points")

        if type(self.material) != Material:
            raise TypeError("argument - material must be Material")

        if type(self.section) != Section:
            raise TypeError("argument - section must be Section")

        if self.extra_point is not None:
            if type(self.extra_point) != list and type(self.extra_point) != np.ndarray:
                raise TypeError("argument - extra_point must be LIST")
            elif len(self.extra_point) != 3:
                raise ValueError("length of list - extra_point must be 3")
            else:
                for element in self.extra_point:
                    if type(element) != float and type(element) != int:
                        raise TypeError("elements of extra_point must be FLOAT or INT")

        if self.direction_vector is not None:
            if type(self.direction_vector) != list and type(self.direction_vector) != np.ndarray:
                raise TypeError("argument - direction_vector must be LIST")
            elif len(self.direction_vector) != 3:
                raise ValueError("length of list - direction_vector must be 3")
            else:
                for element in self.direction_vector:
                    if type(element) != float and type(element) != int:
                        raise TypeError("elements of direction_vector must be FLOAT or INT")
