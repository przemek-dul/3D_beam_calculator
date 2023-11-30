import numpy as np


class Element:
    def __init__(self, node1, node2, material, section):
        self.node1 = node1
        self.node2 = node2
        self.L = np.sqrt(pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2) + pow(node1.z - node2.z, 2))
        self.material = material
        self.section = section

        self.G = self.material.E / (2 * (1 + self.material.v))
        self.phi_z = 12 * self.material.E * self.section.Iz / (self.section.A * self.G * pow(self.L, 2))
        self.phi_y = 12 * self.material.E * self.section.Iy / (self.section.A * self.G * pow(self.L, 2))

        self.t_matrix = None
        self.local_c_matrix = None

        self.S1 = None
        self.T1 = None
        self.S2 = None
        self.T2 = None
        self.S3 = None
        self.T3 = None

        self.von_Mises_stress = None

        self.F_S = None
        self.F_T = None
        self.M_Y = None
        self.F_Y = None
        self.M_Z = None
        self.F_Z = None

        self.calculate_transformation_matrix()

    def get_k_vector(self):
        t_vector = np.array([self.node2.x - self.node1.x,
                             self.node2.y - self.node1.y,
                             self.node2.z - self.node1.z])

        v_vector = np.array([1, t_vector[1], t_vector[2]])
        n2_direction = np.cross(t_vector, v_vector)
        n1_direction = np.cross(n2_direction, t_vector)
        if n1_direction.all() == 0:
            v_vector = np.array([t_vector[0], 1, t_vector[2]])
            n2_direction = np.cross(t_vector, v_vector)
            n1_direction = np.cross(n2_direction, t_vector)
        if n1_direction.all() == 0:
            v_vector = np.array([t_vector[0], t_vector[1], 1])
            n2_direction = np.cross(t_vector, v_vector)
            n1_direction = np.cross(n2_direction, t_vector)

        return n1_direction

    def get_local_k_matrix(self):
        L = self.L
        E = self.material.E
        G = self.G
        Iz = self.section.Iz
        Iy = self.section.Iy
        J = Iz + Iy
        A = self.section.A
        fz = self.phi_z
        fy = self.phi_y

        k_matrix = np.matrix([[E * A / L, 0, 0, 0, 0, 0, -E * A / L, 0, 0, 0, 0, 0],
                              [0, 12 * E * Iz / ((1 + fz) * pow(L, 3)), 0, 0, 0, 6 * E * Iz / ((1 + fz) * pow(L, 2)), 0,
                               -12 * E * Iz / ((1 + fz) * pow(L, 3)), 0, 0, 0, 6 * E * Iz / ((1 + fz) * pow(L, 2))],
                              [0, 0, 12 * E * Iy / ((1 + fy) * pow(L, 3)), 0, -6 * E * Iy / ((1 + fy) * pow(L, 2)), 0,
                               0, 0, -12 * E * Iy / ((1 + fy) * pow(L, 3)), 0, -6 * E * Iy / ((1 + fy) * pow(L, 2)), 0],
                              [0, 0, 0, G * J / L, 0, 0, 0, 0, 0, -G * J / L, 0, 0],
                              [0, 0, -6 * E * Iy / ((1 + fy) * pow(L, 2)), 0, (4 + fy) * E * Iy / ((1 + fy) * L), 0, 0,
                               0, 6 * E * Iy / ((1 + fy) * pow(L, 2)), 0, (2 - fy) * E * Iy / ((1 + fy) * L), 0],
                              [0, 6 * E * Iz / ((1 + fz) * pow(L, 2)), 0, 0, 0, (4 + fz) * E * Iz / ((1 + fz) * L), 0,
                               -6 * E * Iz / ((1 + fz) * pow(L, 2)), 0, 0, 0, (2 - fz) * E * Iz / ((1 + fz) * L)],

                              [-E * A / L, 0, 0, 0, 0, 0, E * A / L, 0, 0, 0, 0, 0],
                              [0, -12 * E * Iz / ((1 + fz) * pow(L, 3)), 0, 0, 0, -6 * E * Iz / ((1 + fz) * pow(L, 2)),
                               0, 12 * E * Iz / ((1 + fz) * pow(L, 3)), 0, 0, 0, -6 * E * Iz / ((1 + fz) * pow(L, 2))],
                              [0, 0, -12 * E * Iy / ((1 + fy) * pow(L, 3)), 0, 6 * E * Iy / ((1 + fy) * pow(L, 2)), 0,
                               0, 0, 12 * E * Iy / ((1 + fy) * pow(L, 3)), 0, 6 * E * Iy / ((1 + fy) * pow(L, 2)), 0],
                              [0, 0, 0, -G * J / L, 0, 0, 0, 0, 0, G * J / L, 0, 0],
                              [0, 0, -6 * E * Iy / ((1 + fy) * pow(L, 2)), 0, (2 - fy) * E * Iy / ((1 + fy) * L), 0, 0,
                               0, 6 * E * Iy / ((1 + fy) * pow(L, 2)), 0, (4 + fy) * E * Iy / ((1 + fy) * L), 0],
                              [0, 6 * E * Iz / ((1 + fz) * pow(L, 2)), 0, 0, 0, (2 - fz) * E * Iz / ((1 + fz) * L), 0,
                               -6 * E * Iz / ((1 + fz) * pow(L, 2)), 0, 0, 0, (4 + fz) * E * Iz / ((1 + fz) * L)]
                              ])
        return k_matrix

    def calculate_transformation_matrix(self):
        t_matrix = np.zeros((12, 12))

        k_vector = self.get_k_vector()
        #k_vector = np.array([0,0,-1])

        T11 = (self.node2.x - self.node1.x) / self.L
        T12 = (self.node2.y - self.node1.y) / self.L
        T13 = (self.node2.z - self.node1.z) / self.L

        A = np.sqrt(pow(T12 * k_vector[2] - T13 * k_vector[1], 2) + pow(T13 * k_vector[0] - T11 * k_vector[2], 2) + pow(T11 * k_vector[1] - T12 * k_vector[0], 2))

        T21 = -(T12 * k_vector[2] - T13 * k_vector[1]) / A
        T22 = -(T13 * k_vector[0] - T11 * k_vector[2]) / A
        T23 = -(T11 * k_vector[1] - T12 * k_vector[0]) / A

        B = np.sqrt(pow(T12*T23-T13*T22, 2) + pow(T13*T21-T11*T23, 2) + pow(T11*T22-T12*T21, 2))

        T31 = (T12*T23 - T13*T22) / B
        T32 = (T13 * T21 - T11 * T23) / B
        T33 = (T11 * T22 - T12 * T21) / B

        t = np.matrix([[T11, T12, T13], [T21, T22, T23], [T31, T32, T33]])
        print(t)
        #t = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        t_matrix[0:3, 0:3] = t
        t_matrix[3:6, 3:6] = t
        t_matrix[6:9, 6:9] = t
        t_matrix[9:12, 9:12] = t



        self.t_matrix = t_matrix

    def get_global_k_matrix(self):
        k_global = self.get_local_k_matrix()

        k_global = self.t_matrix.dot(k_global).dot(self.t_matrix.transpose())

        return k_global

    def shape_func(self, x, direction):
        phi = 0
        if direction == 'uy':
            phi = self.phi_z
        elif direction == 'uz':
            phi = self.phi_y

        x = x / self.L

        N1 = (1 / (1 + phi)) * (1 - 3 * x ** 2 + 2 * x ** 3 + phi * (1 - x))
        N2 = self.L * (1 / (1 + phi)) * (x - 2 * x ** 2 + x ** 3 + phi * (x - x ** 2) / 2)
        N3 = (1 / (1 + phi)) * (3 * x ** 2 - 2 * x ** 3 + phi * x)
        N4 = self.L * (1 / (1 + phi)) * (-x ** 2 + x ** 3 + phi * (-x + x ** 2) / 2)

        N5 = (6*(1 / (1 + phi)) / self.L) * (-x+x**2)
        N6 = (1 / (1 + phi)) * (1-4*x+3*x**2+phi*(1-x))
        N7 = -N5
        N8 = (1 / (1 + phi)) * (-2*x+3*x**2+phi*x)

        return N1, N2, N3, N4, N5, N6, N7, N8

    def calculate_stress(self):
        c_matrix = self.local_c_matrix
        self.S1 = self.material.E * (c_matrix[6, 0] - c_matrix[0, 0]) / self.L
        self.T1 = self.G * (c_matrix[8, 0] - c_matrix[3, 0]) * max(self.section.max_y, self.section.max_z)
        self.S2 = self.material.E * self.section.max_y * (c_matrix[11, 0] - c_matrix[5, 0]) / self.L
        self.T2 = (-self.G * self.phi_z * (1 / (1 + self.phi_z)) * (2 * c_matrix[1, 0] + c_matrix[5, 0] * self.L
                    - 2 * c_matrix[7, 0] + c_matrix[11, 0] * self.L)) / (2 * self.L)
        self.S3 = self.material.E * self.section.max_z * (c_matrix[10, 0] - c_matrix[4, 0]) / self.L
        self.T3 = (-self.G * self.phi_y * (1 / (1 + self.phi_y)) * (2 * c_matrix[2, 0] + c_matrix[4, 0] * self.L
                    - 2 * c_matrix[8, 0] + c_matrix[10, 0] * self.L)) / (2 * self.L)

        self.von_Mises_stress = np.sqrt(0.5*(self.S1**2+3*self.T1**2+self.S2**2+3*self.T2**2+self.S3**2+3*self.T3**2))

    def calculate_forces(self):
        self.F_S = round(self.section.A * self.S1, 5)
        self.F_T = None
        self.M_Y = round(self.S2 * self.section.Iz / self.section.max_y, 5)
        self.F_Y = round(self.T2 * self.section.Iz * self.section.y_wth / self.section.Qz, 5)
        self.M_Z = round(self.S3 * self.section.Iy / self.section.max_z, 5)
        self.F_Z = round(self.T3 * self.section.Iy * self.section.z_wth / self.section.Qy, 5)

    def calculate_local_c_matrix(self):
        c_matrix = np.matrix(
            [[self.node1.ux], [self.node1.uy], [self.node1.uz], [self.node1.fx], [self.node1.fy], [self.node1.fz],
             [self.node2.ux], [self.node2.uy], [self.node2.uz], [self.node2.fx], [self.node2.fy], [self.node2.fz]])
        Q = np.linalg.inv(self.t_matrix)

        self.local_c_matrix = np.dot(Q, c_matrix)

