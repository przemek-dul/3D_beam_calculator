import numpy as np


class Element:
    def __init__(self, node1, node2, material, section, direction_vector, analytical_shear_stresses=False):
        self.node1 = node1
        self.node2 = node2
        self.L = np.sqrt(pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2) + pow(node1.z - node2.z, 2))
        self.material = material
        self.section = section
        self.direction_vector = direction_vector  # direction vector of local z axis
        self.analytical_shear_stresses = analytical_shear_stresses

        self.G = self.material.E / (2 * (1 + self.material.v))  # shear modul
        self.shear_factor = self.section.shear_factor_fun(self.material.v)  # shear correction factor

        self.phi_z = 12 * self.material.E * self.section.Iz / (self.shear_factor * self.section.A * self.G * pow(self.L, 2))
        self.phi_y = 12 * self.material.E * self.section.Iy / (self.shear_factor * self.section.A * self.G * pow(self.L, 2))

        self.t_matrix = None  # transformation matrix to get from local to global system and opposite
        self.local_c_matrix = None   # matrix of local deformations

        self.stress_vector = None
        self.force_vector = None

        self.calculate_transformation_matrix()

    def get_local_k_matrix(self):
        #  returns local stiffness matrix
        L = self.L
        E = self.material.E
        G = self.G
        Iz = self.section.Iz
        Iy = self.section.Iy
        J = Iz + Iy
        A = self.section.A
        fz = self.phi_z
        fy = self.phi_y

        # stiffness matrix form according to the Timoshenko theory
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
                               -6 * E * Iz / ((1 + fz) * pow(L, 2)), 0, 0, 0, (4 + fz) * E * Iz / ((1 + fz) * L)]])

        return k_matrix

    def calculate_transformation_matrix(self):
        t_matrix = np.zeros((12, 12))

        k_vector = self.direction_vector

        T11 = (self.node2.x - self.node1.x) / self.L
        T12 = (self.node2.y - self.node1.y) / self.L
        T13 = (self.node2.z - self.node1.z) / self.L

        A = np.sqrt(pow(T12 * k_vector[2] - T13 * k_vector[1], 2) + pow(T13 * k_vector[0] - T11 * k_vector[2], 2) +
                    pow(T11 * k_vector[1] - T12 * k_vector[0], 2))

        T21 = (T12 * k_vector[2] - T13 * k_vector[1]) / A
        T22 = (T13 * k_vector[0] - T11 * k_vector[2]) / A
        T23 = (T11 * k_vector[1] - T12 * k_vector[0]) / A

        B = np.sqrt(pow(T12*T23-T13*T22, 2) + pow(T13*T21-T11*T23, 2) + pow(T11*T22-T12*T21, 2))

        T31 = -(T12*T23 - T13*T22) / B
        T32 = -(T13 * T21 - T11 * T23) / B
        T33 = (T11 * T22 - T12 * T21) / B

        t = np.matrix([[T11, T12, T13], [T21, T22, T23], [T31, T32, T33]])

        t_matrix[0:3, 0:3] = t
        t_matrix[3:6, 3:6] = t
        t_matrix[6:9, 6:9] = t
        t_matrix[9:12, 9:12] = t

        self.t_matrix = t_matrix

    def get_global_k_matrix(self):
        # returns global stiffness matrix
        k_global = self.get_local_k_matrix()
        Q_n = np.linalg.inv(self.t_matrix)
        k_global = Q_n.dot(k_global).dot(Q_n.transpose())

        return k_global

    def shape_func(self, x, direction):
        # calculation of shape function coefficients - according to the Timoshenko theory
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

    def get_disp_vector(self, resolution, local=False, index=0):
        # returns local or global deformations along element using shape functions
        x_v = np.linspace(0, self.L, resolution)
        c_matrix = self.calculate_local_c_matrix(index)

        # coefficients for approximate longitudinal deformations and rotation on local X axis
        N9 = (c_matrix[6, 0] - c_matrix[0, 0]) / self.L
        N10 = c_matrix[0, 0]

        N11 = (c_matrix[9, 0] - c_matrix[3, 0]) / self.L
        N12 = c_matrix[3, 0]

        disp_matrix = np.empty((8, 0))
        local_disp_matrix = np.empty((6, 0))

        for x in x_v:
            # local Z direction and local Y axis
            N1, N2, N3, N4, N5, N6, N7, N8 = self.shape_func(x, 'uy')
            u_yy = N1 * c_matrix[1, 0] + N2 * c_matrix[5, 0] + N3 * c_matrix[7, 0] + N4 * c_matrix[11, 0]
            f_zz = N5 * c_matrix[1, 0] + N6 * c_matrix[5, 0] + N7 * c_matrix[7, 0] + N8 * c_matrix[11, 0]

            # local Y direction and local Z axis
            N1, N2, N3, N4, N5, N6, N7, N8 = self.shape_func(x, 'uz')
            u_zz = N1 * c_matrix[2, 0] - N2 * c_matrix[4, 0] + N3 * c_matrix[8, 0] - N4 * c_matrix[10, 0]
            f_yy = N5 * c_matrix[2, 0] - N6 * c_matrix[4, 0] + N7 * c_matrix[8, 0] - N8 * c_matrix[10, 0]

            #  linear functions for deformation in local X direction and rotation in local X axis
            u_xx = N9 * x + N10
            f_xx = N11 * x + N12

            # transformation of calculated local deformations to global system
            l_matrix = np.matrix([[u_xx], [u_yy], [u_zz], [f_xx], [f_yy], [f_zz]])
            g_matrix = np.dot(np.linalg.inv(self.t_matrix[0:6, 0:6]), l_matrix)

            total_disp = np.sqrt(pow(g_matrix[0, 0], 2) + pow(g_matrix[1, 0], 2) + pow(g_matrix[2, 0], 2))
            total_rot = np.sqrt(pow(g_matrix[3, 0], 2) + pow(g_matrix[4, 0], 2) + pow(g_matrix[5, 0], 2))

            g_matrix = np.vstack((g_matrix, [total_disp]))
            g_matrix = np.vstack((g_matrix, [total_rot]))

            g_matrix = np.array(g_matrix.transpose()[0])

            disp_matrix = np.append(disp_matrix, g_matrix.transpose(), axis=1)
            local_disp_matrix = np.append(local_disp_matrix, np.array(l_matrix), axis=1)

        #  coordinates of points for which deformations was calculated
        points = np.array(
            [np.linspace(self.node1.points_vector[i], self.node2.points_vector[i], resolution) for i in range(0, 3)])

        if not local:
            return disp_matrix, points
        else:
            return local_disp_matrix, points

    def get_stress_force_vector(self, resolution, index=0):
        stress_vector = np.empty((6, 0))
        force_vector = np.empty((6, 0))

        # local deformations to calculate stress along element
        disp, points = self.get_disp_vector(resolution, local=True, index=index)

        for i in range(0, len(disp[0]) - 1):
            l = np.sqrt(pow(points[0, i + 1] - points[0, i], 2) + pow(points[1, i + 1] - points[1, i], 2) + pow(
                points[2, i + 1] - points[2, i], 2))

            #  Normal stress due to stretch
            S1 = round(self.material.E * (disp[0, i + 1] - disp[0, i]) / l, 5)

            #  Maximum shear stress due to torsion
            T1 = round(self.G * (disp[3, i + 1] - disp[3, i]) * max(self.section.max_y, self.section.max_z) / l, 5)

            #  Maximum Normal stress due to bending in y-direction
            S2 = round(self.material.E * self.section.max_y * (disp[5, i + 1] - disp[5, i]) / l, 5)

            #  Shear stress due to bending in y-direction
            T2 = round((-self.G * self.phi_z * (1 / (1 + self.phi_z)) * (
                        2 * disp[1, i] + disp[5, i] * l - 2 * disp[1, i + 1] + disp[5, i + 1] * l)) / (2 * l), 5)

            #  Maximum Normal stress due to bending in z-direction
            S3 = round(self.material.E * self.section.max_z * (disp[4, i + 1] - disp[4, i]) / l, 5)

            #  Shear stress due to bending in z-direction
            T3 = round((-self.G * self.phi_y * (1 / (1 + self.phi_y)) * (
                        2 * disp[2, i] + disp[4, i] * l - 2 * disp[2, i + 1] + disp[4, i + 1] * l)) / (2 * l), 5)

            #  Forces based on stress and properties of cross-section
            F_X = round(self.section.A * S1, 5)
            M_X = round(T1 * (self.section.Iz + self.section.Iy) / max(self.section.max_z, self.section.max_y), 5)
            M_Z = round(S2 * self.section.Iz / self.section.max_y, 5)
            F_Y = round(T2 * self.section.A * self.shear_factor, 5)
            M_Y = round(S3 * self.section.Iy / self.section.max_z, 5)
            F_Z = round(T3 * self.section.A * self.shear_factor, 5)

            if self.analytical_shear_stresses:
                if self.section.custom:
                    raise AttributeError(
                        "Analytical_shear_stresses stress is available only for standard cross sections")

                T1 = self.section.torsion_shear(M_X, self.section.max_z, 0)
                T2 = F_Y * self.section.bending_shear(np.array([[0]]), 'z') / self.section.Iz
                T3 = F_Z * self.section.bending_shear(np.array([[0]]), 'y') / self.section.Iy
                T2 = T2[0,0]
                T3 = T3[0,0]

            bufor_stress = np.array([[S1], [T2], [T3], [T1], [S2], [S3]])

            bufor_force = np.array([[F_X], [F_Y], [F_Z], [M_X], [M_Y], [M_Z]])

            stress_vector = np.append(stress_vector, bufor_stress, axis=1)
            force_vector = np.append(force_vector, bufor_force, axis=1)

        return stress_vector, force_vector

    def get_max_displacements(self, resolution, index=0, local=False):
        disp_vector, points = self.get_disp_vector(resolution, index=index, local=local)
        max_vector = np.array([[np.max(np.abs(disp))] for disp in disp_vector])

        return max_vector

    def get_max_stress(self, resolution, index=0):
        stress_vector, force_vector = self.get_stress_force_vector(resolution, index)
        max_vector = np.array([[np.max(np.abs(stress))] for stress in stress_vector])

        return max_vector

    def get_max_force(self, resolution, index=0):
        stress_vector, force_vector = self.get_stress_force_vector(resolution, index)
        max_vector = np.array([[np.max(np.abs(force))] for force in force_vector])

        return max_vector

    def get_residuals(self, index=0):
        # returns residuals forces at nodes of element
        stress_vector, force_vector = self.get_stress_force_vector(resolution=150, index=index)

        node1 = -force_vector[:, 0].dot(np.linalg.inv(self.t_matrix))
        node2 = -force_vector[:, -1].dot(np.linalg.inv(self.t_matrix))

        return {self.node1.index: node1, self.node2.index: node2}

    def calculate_local_c_matrix(self, index=0):
        #  calculation of local deformation at nodes based on model results
        #  method called after solving linear equations in the analysis model
        c_matrix = np.matrix(
            [[self.node1.displacement_vector[index, 0]], [self.node1.displacement_vector[0, 1]],
             [self.node1.displacement_vector[index, 2]], [self.node1.displacement_vector[0, 3]],
             [self.node1.displacement_vector[index, 4]], [self.node1.displacement_vector[0, 5]],
             [self.node2.displacement_vector[index, 0]], [self.node2.displacement_vector[0, 1]],
             [self.node2.displacement_vector[index, 2]], [self.node2.displacement_vector[0, 3]],
             [self.node2.displacement_vector[index, 4]], [self.node2.displacement_vector[0, 5]]])

        return np.dot(self.t_matrix, c_matrix)

    def get_section_stresses(self, length, resolution, index=0):
        # returns stress distribution on the cross-section at specified length
        if self.section.custom:
            raise AttributeError("Section stress is available only for standard cross sections")

        _, forces = self.get_stress_force_vector(resolution, index)
        step = self.L / (resolution-2)
        ind = int(length / step)
        A = self.section.A
        z = self.section.z_points
        y = self.section.y_points
        Iz = self.section.Iz
        Iy = self.section.Iy

        #  Normal stress due to stretch
        S1 = (forces[0, ind] / A) * np.ones(np.shape(z))

        #  Shear stress due to torsion
        T1 = self.section.torsion_shear(forces[3, ind], z, y)

        #  Normal stress due to bending in y-direction
        S2 = forces[5, ind] * y / Iz

        #  Shear stress due to bending in y-direction
        T2 = forces[1, ind] * self.section.bending_shear(y, 'z') / Iz

        #  Normal stress due to bending in z-direction
        S3 = forces[4, ind] * z / Iy

        #  Shear stress due to bending in z-direction
        T3 = forces[2, ind] * self.section.bending_shear(z, 'y') / Iy

        vMs = np.sqrt((S1 + S2 + S3) ** 2 + 3 * (T1 ** 2 + T2 ** 2 + T3 ** 2))

        stress_vector = np.array([S1, T2, T3, T1, S2, S3, vMs])

        stress_vector = np.array([self.section.mask(stress) for stress in stress_vector])

        return stress_vector

    def get_vMs(self, resolution, index=0):
        #  returns maximum von Misses stress along element based on cross section stress distribution
        if self.section.custom:
            raise AttributeError("Von Misses stress stress is available only for standard cross sections")
        lengths = np.linspace(0, self.L, resolution-1)
        vMs = np.array([round(np.amax(self.get_section_stresses(l, resolution, index)), 5) for l in lengths])
        return vMs

