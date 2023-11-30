import numpy as np


class Force:
    def __init__(self, point, direction, value):
        self.type = 'Force'
        self.direction = direction.lower()
        self.value = value
        self.point = point
        self.node = None

    def get_node(self):
        if self.direction == 'x':
            node = 6 * (self.point.node_number - 1)
        elif self.direction == 'y':
            node = 6 * (self.point.node_number - 1) + 1
        elif self.direction == 'z':
            node = 6 * (self.point.node_number - 1) + 2
        return node


class Torque:
    def __init__(self, point, axis, value):
        self.type = 'Torque'
        self.axis = axis.lower()
        self.value = value
        self.point = point
        self.node = None

    def get_node(self):
        if self.axis == 'x':
            node = 6 * (self.point.node_number - 1) + 3
        elif self.axis == 'y':
            node = 6 * (self.point.node_number - 1) + 4
        elif self.axis == 'z':
            node = 6 * (self.point.node_number - 1) + 5
        return node


class Pressure:
    def __init__(self, line, mode, value):
        self.type = 'Pressure'
        self.line = line
        self.value = value
        self.mode = mode

    def get_nodes(self):
        if self.mode == 1:
            values = []
            for element in self.line.elements:
                n1 = self.value*element.L / 2
                n2 = self.value*pow(element.L, 2) / 12

                matrix = np.matrix([[0], [n1], [0], [0], [0], [n2], [0], [n1], [0], [0], [0], [-n2]])
                Q = np.linalg.inv(element.t_matrix)
                matrix = np.dot(Q, matrix)

                for n in range(0, 6):
                    values.append((6 * (element.node1.index - 1) + n, matrix[n, 0]))
                    values.append((6 * (element.node2.index - 1) + n, matrix[n+6, 0]))
            return values


class Displacement:
    def __init__(self, point, ux=None, uy=None, uz=None, fx=None, fy=None, fz=None, DOF=False):
        self.type = 'displacement'
        self.point = point
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.DOF = DOF

        self.check_dof()

    def check_dof(self):
        if self.DOF:
            self.ux = 0
            self.uy = 0
            self.uz = 0
            self.fx = 0
            self.fy = 0
            self.fz = 0

    def get_nodes(self):
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
        if self.fx is not None:
            n = (6 * (self.point.node_number - 1) + 3, self.fx)
            output.append(n)
        if self.fy is not None:
            n = (6 * (self.point.node_number - 1) + 4, self.fy)
            output.append(n)
        if self.fz is not None:
            n = (6 * (self.point.node_number - 1) + 5, self.fz)
            output.append(n)

        return output

