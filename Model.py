import numpy as np
from loguru import logger
from Mesh import Mesh
from Load import Displacement, Force, Torque, Pressure


class Static:
    def __init__(self, mesh: Mesh, displacement_bc: list, forces_bc: list):
        self.mesh = mesh
        self.displacement_bc = displacement_bc  # boundary conditions of the restraint
        self.forces_bc = forces_bc  # forcing boundary conditions

        self.check_input()
        self.check_if_fix()

        self.elements = self.mesh.elements
        self.nodes = self.mesh.nodes
        self.current_node = self.mesh.current_node

        self.displacement_points = []  # points where restraint boundary condition was added

        self.k_matrix = None  # global stiffness matrix
        self.c_matrix = None  # global force matrix
        self.x_matrix = None  # global deformations matrix

        self.solved = False

    def check_input(self):
        if type(self.mesh) != Mesh:
            raise TypeError("argument - mesh must be Mesh")

        if type(self.displacement_bc) != list and type(self.displacement_bc) != np.ndarray:
            raise TypeError("argument - displacement_bc must be LIST")
        else:
            for bc in self.displacement_bc:
                if type(bc) != Displacement:
                    raise TypeError("Elements of displacement_bc list must be Displacement")

        if type(self.forces_bc) != list and type(self.forces_bc) != np.ndarray:
            raise TypeError("argument forces_BC must be List")
        else:
            for bc in self.forces_bc:
                if type(bc) != Force and type(bc) != Torque and type(bc) != Pressure:
                    raise TypeError("elements of forces_bc list must be Force, Torque or Pressure")

    def check_if_fix(self):
        # check if system is fixed in space
        ux = False
        uy = False
        uz = False
        nx = 0
        ny = 0
        nz = 0
        rotx = False
        roty = False
        rotz = False

        for load in self.displacement_bc:
            if load.ux is not None:
                ux = True
                nx += 1
            if load.uy is not None:
                uy = True
                ny += 1
            if load.uz is not None:
                uz = True
                nz += 1
            if load.rot_x is not None:
                rotx = True
                nz += 1
            if load.rot_y is not None:
                roty = True
                nz += 1
            if load.rot_z is not None:
                rotz = True
                nz += 1

        if nx > 1 or ny > 1:
            rotz = True
        if nx > 1 or nz > 1:
            roty = True
        if ny > 1 or ny > 1:
            rotx = True

        if not(ux and uy and uz and rotx and roty and rotz):
            raise AttributeError('system is not fixed - check boundary conditions')

    def create_k_matrix(self):
        # calculation of global stiffness matrix, by add all global stiffness matrix of mesh elements
        logger.warning('Creating global stiffness matrix..')

        # creation of empty matrix to start adding - shape determined based on number of elements in the mesh
        self.k_matrix = np.zeros((6*self.current_node, 6*self.current_node))
        for element in self.elements:
            k_local = element.get_global_k_matrix()

            q1 = k_local[0:6, 0:6]
            q2 = k_local[0:6, 6:12]
            q3 = k_local[6:12, 0:6]
            q4 = k_local[6:12, 6:12]
            """
            adding quarters of element global stiffness matrix to the model stiffness matrix,
            based on position of the element in the mesh
            """
            self.k_matrix[6 * (element.node1.index - 1):6 * (element.node1.index - 1) + 6,
            6 * (element.node1.index - 1):6 * (element.node1.index - 1) + 6] += q1
            self.k_matrix[6 * (element.node1.index - 1):6 * (element.node1.index - 1) + 6,
            6 * (element.node2.index - 1):6 * (element.node2.index - 1) + 6] += q2
            self.k_matrix[6 * (element.node2.index - 1):6 * (element.node2.index - 1) + 6,
            6 * (element.node1.index - 1):6 * (element.node1.index - 1) + 6] += q3
            self.k_matrix[6 * (element.node2.index - 1):6 * (element.node2.index - 1) + 6,
            6 * (element.node2.index - 1):6 * (element.node2.index - 1) + 6] += q4

        logger.info("Global stiffness matrix created")

    def apply_loads(self):

        logger.warning("Applying loads...")

        self.c_matrix = np.matrix(np.zeros((self.current_node*6, 1)))

        for load in self.forces_bc:
            if load.type == 'Force' or load.type == 'Torque':
                if load.point.node_number is not None:
                    # adding force or moment boundary condition to the global force matrix
                    self.c_matrix[load.get_node(), 0] += load.value
                else:
                    raise AttributeError(f"argument - point for Force and Torque must be meshed")

            elif load.type == 'Pressure':
                # Distributed load

                # check if input line is meshed
                if len(load.line.elements_index) != 0:
                    value_vector = load.value

                    if type(value_vector) == float or type(value_vector) == int:
                        value_vector = value_vector * np.ones(len(load.line.elements_index) * 10)
                    else:
                        load.extend_value_vector()
                        value_vector = load.value

                    line = load.line
                    x_vector = np.linspace(0, line.len, len(value_vector))  # points on the line

                    # x1, x2 - local coordinates on the line at the beginning and at the end of element
                    x1 = 0
                    for index in load.line.elements_index:
                        element = self.elements[index]
                        x2 = x1 + element.L

                        """
                        Indexes for find pressure value at the beginning and at the end of the element.
                        If 'id' is single array, boundary values exist in input 'value' array. If 'id' is double array, 
                        boundary values are approximated ba linear function based of adjacent values, located by
                        returned indexes. 
                        """

                        id_1 = load.find_index(x1, x_vector)
                        id_2 = load.find_index(x2, x_vector)

                        # pressure vector acting at element
                        in_values = value_vector[id_1[-1]:id_2[0]+1]

                        # approximation of pressure at the beginning of element, using linear function if needed
                        if len(id_1) == 2:
                            a = (value_vector[id_1[1]] - value_vector[id_1[0]]) / (x_vector[id_1[1]] - x_vector[id_1[0]])
                            b = value_vector[id_1[0]] - a * x_vector[id_1[0]]

                            to_add = x1 * a + b
                            in_values = np.insert(in_values, 0, to_add)

                        # approximation of pressure at the end of element, using linear function if needed
                        if len(id_2) == 2:
                            a = (value_vector[id_2[1]] - value_vector[id_2[0]]) / (x_vector[id_2[1]] - x_vector[id_2[0]])
                            b = value_vector[id_2[0]] - a * x_vector[id_2[0]]

                            to_add = x2 * a + b
                            in_values = np.append(in_values, to_add)

                        x1 = x2

                        in_x = np.linspace(0, element.L, len(in_values))

                        # calculation of coefficients to calculate the reactions caused by distributed load
                        n1_vector = np.array([load.calc_N1(x, element.L) for x in in_x])
                        n2_vector = np.array([load.calc_N2(x, element.L) for x in in_x])
                        n3_vector = np.array([load.calc_N3(x, element.L) for x in in_x])
                        n4_vector = np.array([load.calc_N4(x, element.L) for x in in_x])

                        dx = in_x[1] - in_x[0]  # distance between points

                        # integrations
                        f1 = np.trapz(n1_vector * in_values, dx=dx)
                        m1 = np.trapz(n2_vector * in_values, dx=dx)
                        f2 = np.trapz(n3_vector * in_values, dx=dx)
                        m2 = np.trapz(n4_vector * in_values, dx=dx)

                        # local force matrix creation
                        if load.direction == 'y':
                            matrix = np.matrix([[0], [f1], [0], [0], [0], [m1], [0], [f2], [0], [0], [0], [m2]])
                        else:
                            matrix = np.matrix([[0], [0], [f1], [0], [m1], [0], [0], [0], [f2], [0], [m2], [0]])

                        # transformation of local force matrix to global force matrix
                        t_matrix = np.linalg.inv(element.t_matrix)
                        matrix = t_matrix.dot(matrix)

                        # adding pressure boundary condition to the global force matrix
                        for n in range(0, 6):
                            self.c_matrix[6 * (element.node1.index - 1) + n, 0] += matrix[n, 0]
                            self.c_matrix[6 * (element.node2.index - 1) + n, 0] += matrix[n + 6, 0]
                else:
                    raise AttributeError(f"argument - line for Pressure bc must be meshed")

        for load in self.displacement_bc:
            if load.point.node_number is not None:
                self.displacement_points.append(load.point)
                nodes = load.get_nodes()
                for node in nodes:
                    # resetting the row of the stiffness matrix to apply the given deformation
                    self.k_matrix[node[0], :] = 0
                    # adding displacement boundary condition to the global force matrix
                    self.c_matrix[node[0], 0] = node[1]
            else:
                raise AttributeError(f"argument - point for displacement bc must be meshed")

        # assigning 1 on the diagonal of the matrix for zero elements to make matrix possible to invert
        for i in range(0, 6 * self.current_node):
            if self.k_matrix[i, i] == 0:
                self.k_matrix[i, i] = 1

        logger.info("Loads applied")

    def solve(self):

        self.create_k_matrix()
        self.apply_loads()

        logger.warning("Solving linear equations...")

        # solving a system of linear equations
        self.x_matrix = np.linalg.solve(self.k_matrix, self.c_matrix)

        for n in range(0, self.current_node):
            for i in range(0, 6):
                self.nodes[n].displacement_vector[0, i] = self.x_matrix[6 * n + i, 0]

        self.solved = True
        logger.info("Solution done")

    def get_elements_disp(self, resolution, index=0):
        # returns 2d array of deformation for all elements
        disp_vector = np.empty((0, 8, resolution))
        points_vector = np.empty((0, 3, resolution))

        for element in self.elements:
            in_disp, in_points = element.get_disp_vector(resolution, index=index)
            disp_vector = np.vstack((disp_vector, [in_disp]))
            points_vector = np.vstack((points_vector, [in_points]))

        return disp_vector, points_vector

    def get_elements_stress_force(self, resolution, index=0):
        # returns 2d arrays of stresses and forces for all elements
        stress_vector = np.empty((0, 7, resolution-1))
        force_vector = np.empty((0, 6, resolution-1))

        for element in self.elements:
            in_stress, in_force = element.get_stress_force_vector(resolution, index=index)

            stress_vector = np.vstack((stress_vector, [in_stress]))
            force_vector = np.vstack((force_vector, [in_force]))

        return stress_vector, force_vector

    def get_points(self):
        # returns all points that creates geometry
        output = np.array([])
        for point in self.mesh.points:
            val = {'index': point.index, 'coordinates': point.point_vector}
            output = np.append(output, val)

        return output
