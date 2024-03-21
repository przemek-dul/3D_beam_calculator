import numpy as np
from Point import Point
from Line import Line


class Force:
    def __init__(self, point: Point, direction: str, value: float):
        self.type = 'Force'
        self.direction = direction  # global direction
        self.value = value
        self.point = point
        self.node = None  # index that specify, where in the model's input matrix pass value of load

        self.check_input()

    def get_node(self):
        #  calculate node index for load application, based on input point and global direction
        node = None
        if self.direction == 'x':
            node = 6 * (self.point.node_number - 1)
        elif self.direction == 'y':
            node = 6 * (self.point.node_number - 1) + 1
        elif self.direction == 'z':
            node = 6 * (self.point.node_number - 1) + 2
        return node

    def check_input(self):
        if type(self.point) != Point:
            raise TypeError('argument point must be Point')
        if type(self.direction) != str:
            raise TypeError('argument - direction must be STRING')
        else:
            self.direction = self.direction.lower()
        if self.direction != 'x' and self.direction != 'y' and self.direction != 'z':
            raise ValueError("argument - direction must take one of the following values:  'x', 'y' or 'z'")
        if type(self.value) != int and type(self.value) != float:
            raise TypeError('argument - value of force must be INT of FLOAT')


class Torque(Force):
    #  Torque load same as Force, only the application index moved by 3
    def __init__(self, point: Point, axis: str, value: float):
        self.axis = axis  # global axis
        super().__init__(point, axis, value)
        self.type = 'Torque'

    def get_node(self):
        return super().get_node() + 3

    def check_input(self):
        if type(self.point) != Point:
            raise TypeError('argument - point of must be Point')
        if type(self.axis) != str:
            raise TypeError('argument - axis of be STRING')
        else:
            self.axis = self.axis.lower()
        if self.axis != 'x' and self.axis != 'y' and self.axis != 'z':
            raise ValueError("argument - axis must take one of the following values:  'x', 'y' or 'z'")
        if type(self.value) != int and type(self.value) != float:
            raise TypeError('argument - value of force must be INT of FLOAT')


class Pressure:
    def __init__(self, line: Line, value: float or list, direction: str):
        self.type = 'Pressure'
        self.line = line  # whole line that distributed load acts on whole input line
        self.value = value  # single value for constant input and vector for non-constant
        self.direction = direction  # local direction of distributed load

        self.check_input()

    def find_index(self, value, values):
        # see model.apply_loads... pressure bc
        for i in range(0, len(values)):
            if value < values[i]:
                return [i, i + 1]
            elif value == values[i]:
                return [i]

    def extend_value_vector(self):
        """
        The density of vectors should be high enough to have at least three values per line element.
        Otherwise, input torque at nodes will take the value 0. Function extend the vector by approximation
        by linear function.
        """
        vector = np.array([])
        if len(self.value) < 5 * len(self.line.elements_index):
            to_extend = 5 * len(self.line.elements_index) - len(self.value)
            steps = int(np.ceil(to_extend / (len(self.value)-1)) + 2)
            for i in range(0, len(self.value)-1):
                in_vector = np.linspace(self.value[i], self.value[i+1], steps)
                if i == 0:
                    vector = np.append(vector, in_vector)
                else:
                    vector = np.append(vector, in_vector[1:])

        self.value = vector

    #  variables used to calculate residuals forces and moments at nodes, acting as a result of distributed load
    def calc_N1(self, x, l):
        return 1 - 3 * x ** 2 / l ** 2 + 2 * x ** 3 / l ** 3

    def calc_N2(self, x, l):
        return x - 2 * x ** 2 / l + x ** 3 / l ** 2

    def calc_N3(self, x, l):
        return 3 * x ** 2 / l ** 2 - 2 * x ** 3 / l ** 3

    def calc_N4(self, x, l):
        return -x ** 2 / l + x ** 3 / l ** 2

    def check_input(self):
        if type(self.line) != Line:
            raise TypeError("argument - line must be Line")

        if type(self.value) != int and type(self.value) != float and type(self.value) != list \
                and type(self.value) != np.ndarray:
            raise TypeError('argument - value must be LIST or FLOAT if const')

        elif type(self.value) == list or type(self.value) == np.ndarray:
            if len(self.value) < 2:
                raise AttributeError("length of list must be greater than 1")
            for element in self.value:
                if type(element) != float and type(element) != int and type(element) != np.float64:
                    print(type(element))
                    raise TypeError('elements of list - value  must be FLOAT')


class Displacement:
    def __init__(self, point: Point, ux: float = None, uy: float = None, uz: float = None, rot_x: float = None,
                 rot_y: float = None, rot_z: float = None, DOF: bool = False):
        self.type = 'displacement'
        self.point = point
        # if variable is equal to None, system is free to move in this direction
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
        self.DOF = DOF  # true if all degrees of freedom are to be taken away

        self.check_input()
        self.check_dof()

    def check_dof(self):
        #  Taking away all degrees of freedom if DOF is equal to True
        if self.DOF:
            self.ux = 0
            self.uy = 0
            self.uz = 0
            self.rot_x = 0
            self.rot_y = 0
            self.rot_z = 0

    def get_nodes(self):
        #  The function returns array, node index of load application and displacement value at this index.
        output = []
        if self.ux is not None:
            n = (6 * (self.point.node_number - 1), self.ux)
            output.append(n)
        if self.uy is not None:
            n = (6 * (self.point.node_number - 1) + 1, self.uy)
            output.append(n)
        if self.uz is not None:
            n = (6 * (self.point.node_number - 1) + 2, self.uz)
            output.append(n)
        if self.rot_x is not None:
            n = (6 * (self.point.node_number - 1) + 3, self.rot_x)
            output.append(n)
        if self.rot_y is not None:
            n = (6 * (self.point.node_number - 1) + 4, self.rot_y)
            output.append(n)
        if self.rot_z is not None:
            n = (6 * (self.point.node_number - 1) + 5, self.rot_z)
            output.append(n)

        return output

    def check_input(self):
        if type(self.point) != Point:
            raise TypeError('argument - point must be Point')
        if type(self.ux) != int and type(self.ux) != float and self.ux is not None:
            raise TypeError('argument - ux must be INT or FLOAT')
        if type(self.uy) != int and type(self.uy) != float and self.uy is not None:
            raise TypeError('argument - uy must be INT or FLOAT')
        if type(self.uz) != int and type(self.uz) != float and self.uz is not None:
            raise TypeError('argument - uz must be INT or FLOAT')
        if type(self.rot_x) != int and type(self.rot_x) != float and self.rot_x is not None:
            raise TypeError('argument - rot_x must be INT or FLOAT')
        if type(self.rot_y) != int and type(self.rot_y) != float and self.rot_y is not None:
            raise TypeError('argument - rot_y must be INT or FLOAT')
        if type(self.rot_z) != int and type(self.rot_z) != float and self.rot_z is not None:
            raise TypeError('argument - rot_z must be INT or FLOAT')
        if type(self.DOF) != bool and self.DOF is not None:
            raise TypeError('argument - DOF must be BOOL or None')
