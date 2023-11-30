from Element import Element
from Node import Node
import numpy as np
from loguru import logger


class Model:
    def __init__(self, geometry, displacement_bc, forces_bc):
        self.geometry = geometry
        self.displacement_bc = displacement_bc
        self.forces_bc = forces_bc
        self.elements = []
        self.nodes = []
        self.current_node = 0
        self.k_matrix = None
        self.c_matrix = None
        self.x_matrix = None

    def mesh(self, el_on_line):

        logger.warning("Meshing...")

        self.current_node = 0
        for line in self.geometry:
            if line.point1.node_number is None:
                self.current_node = self.current_node + 1
                line.point1.node_number = self.current_node
                node1 = Node(x=line.point1.x, y=line.point1.y, z=line.point1.z, index=self.current_node)
                self.nodes.append(node1)
            else:
                node1 = self.nodes[line.point1.node_number-1]
            for n in range(1, el_on_line):
                self.current_node = self.current_node + 1

                x = line.point1.x + (n / el_on_line) * (line.point2.x - line.point1.x)
                y = line.point1.y + (n / el_on_line) * (line.point2.y - line.point1.y)
                z = line.point1.z + (n / el_on_line) * (line.point2.z - line.point1.z)

                node2 = Node(x=x, y=y, z=z, index=self.current_node)
                self.nodes.append(node2)
                element = Element(node1, node2, line.material, line.section)
                line.elements.append(element)
                self.elements.append(element)

                node1 = node2

            if line.point2.node_number is None:
                self.current_node = self.current_node + 1
                line.point2.node_number = self.current_node
                node2 = Node(x=line.point2.x, y=line.point2.y, z=line.point2.z, index=self.current_node)
                self.nodes.append(node2)
            else:
                node2 = self.nodes[line.point2.node_number-1]

            element = Element(node1, node2, line.material, line.section)
            line.elements.append(element)
            self.elements.append(element)

        logger.info("Meshing done")

        self.create_k_matrix()
        self.apply_loads()

    def create_k_matrix(self):

        logger.warning('Creating global stiffness matrix..')

        self.k_matrix = np.zeros((6*self.current_node, 6*self.current_node))
        for element in self.elements:
            k_local = element.get_global_k_matrix()

            q1 = k_local[0:6, 0:6]
            q2 = k_local[0:6, 6:12]
            q3 = k_local[6:12, 0:6]
            q4 = k_local[6:12, 6:12]

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

        for load in self.displacement_bc:
            nodes = load.get_nodes()
            for node in nodes:
                self.k_matrix[node[0], :] = 0
                self.c_matrix[node[0], 0] = node[1]

        for load in self.forces_bc:
            if load.type == 'Force' or load.type == 'Torque':
                self.c_matrix[load.get_node(), 0] = load.value
            elif load.type == 'Pressure':
                nodes = load.get_nodes()
                for node in nodes:
                    self.c_matrix[node[0], 0] = self.c_matrix[node[0], 0] + node[1]

        for i in range(0, 6 * self.current_node):
            if self.k_matrix[i, i] == 0:
                self.k_matrix[i, i] = 1

        logger.info("Loads applied")

    def solve(self):

        logger.warning("Solving...")

        self.x_matrix = np.linalg.solve(self.k_matrix, self.c_matrix)
        for n in range(0, self.current_node):
            self.nodes[n].ux = self.x_matrix[6*n, 0]
            self.nodes[n].uy = self.x_matrix[6 * n + 1, 0]
            self.nodes[n].uz = self.x_matrix[6 * n + 2, 0]
            self.nodes[n].fx = self.x_matrix[6 * n + 3, 0]
            self.nodes[n].fy = self.x_matrix[6 * n + 4, 0]
            self.nodes[n].fz = self.x_matrix[6 * n + 5, 0]

        for element in self.elements:
            element.calculate_local_c_matrix()
            element.calculate_stress()
            element.calculate_forces()

        logger.info("Solution done")

    def mesh_info(self):
        return {'Quantity of elements': len(self.elements), 'Quantity of nodes': len(self.nodes)}


