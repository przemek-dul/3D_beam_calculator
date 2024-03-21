from modules.Element import Element
from modules.Node import Node
from Line import Line
import numpy as np


class Mesh:
    def __init__(self):
        self.current_node = 0  # last created node index
        self.nodes = []
        self.elements = []
        self.points = []
        self.lines = []

    def elements_on_line(self, num: int, lines: list):
        # divides lines for 'num' elements of equal length
        if type(num) != int:
            raise TypeError("argument - num must be INT")
        elif num <= 0:
            raise ValueError("argument - num must be greater than 0")
        if type(lines) != list and type(lines) != np.ndarray:
            raise TypeError("argument - Lines must be LIST")
        else:
            for line in lines:
                if type(line) != Line:
                    raise TypeError("elements of lines must be Line")

        for line in lines:
            # first node creation at the beginning of line - coordinates of node same as first point of line
            if line.point1.node_number is None:
                self.current_node = self.current_node + 1
                line.point1.node_number = self.current_node
                self.lines.append(line)
                node1 = Node(x=line.point1.x, y=line.point1.y, z=line.point1.z, index=self.current_node)
                self.nodes.append(node1)
                self.points.append(line.point1)
            else:
                # Condition if first node was already created on another line
                node1 = self.nodes[line.point1.node_number - 1]
            # dividing line for 'num' elements
            for n in range(1, num):
                self.current_node = self.current_node + 1
                #  distance between nodes at specify direction is equal to (n / num) * (line.point2.x - line.point1.x)
                #  coordinates of new node
                x = line.point1.x + (n / num) * (line.point2.x - line.point1.x)
                y = line.point1.y + (n / num) * (line.point2.y - line.point1.y)
                z = line.point1.z + (n / num) * (line.point2.z - line.point1.z)

                # creation new node and element based on two last created nodes
                node2 = Node(x=x, y=y, z=z, index=self.current_node)
                self.nodes.append(node2)
                element = Element(node1, node2, line.material, line.section, line.direction_vector)

                line.elements_index.append(len(self.elements))
                self.lines.append(line)
                self.elements.append(element)
                element.node1.elements_index.append(len(self.elements)-1)
                element.node2.elements_index.append(len(self.elements)-1)
                node1 = node2

            if line.point2.node_number is None:
                # last node creation at the end of the line - coordinates of node same as last point of line
                self.current_node = self.current_node + 1
                line.point2.node_number = self.current_node
                self.lines.append(line)
                node2 = Node(x=line.point2.x, y=line.point2.y, z=line.point2.z, index=self.current_node)
                self.nodes.append(node2)
                self.points.append(line.point2)
            else:
                # Condition if last node was already created on another line
                node2 = self.nodes[line.point2.node_number - 1]

            # creation last element on the line
            element = Element(node1, node2, line.material, line.section, line.direction_vector)
            line.elements_index.append(len(self.elements))
            self.lines.append(line)
            self.elements.append(element)
            element.node1.elements_index.append(len(self.elements) - 1)
            element.node2.elements_index.append(len(self.elements) - 1)

    def max_element_size(self, size: float, lines: list):
        # divides lines for elements with length equals to 'size'
        if type(size) != float and type(size) != int:
            raise TypeError("argument - size must be FLOAT or INT")
        elif size <= 0:
            raise ValueError("argument - size must be greater than 0")
        if type(lines) != list and type(lines) != np.ndarray:
            raise TypeError("argument - lines must be LIST")
        else:
            for line in lines:
                if type(line) != Line:
                    raise TypeError("elements of lines must be Line")

        for line in lines:
            # first node creation, at the beginning of line - coordinates of node same as first point of line
            if line.point1.node_number is None:
                self.current_node = self.current_node + 1
                line.point1.node_number = self.current_node
                self.lines.append(line)
                node1 = Node(x=line.point1.x, y=line.point1.y, z=line.point1.z, index=self.current_node)
                self.nodes.append(node1)
                self.points.append(line.point1)
            else:
                # Condition if first node was already created on another line
                node1 = self.nodes[line.point1.node_number - 1]

            # if input size is bigger than line length, method will create one element on the line
            if line.len > size:
                # calculation of distance between nodes

                # calculation of proportions of how the length of the element affects each axis
                dx = line.point2.x - line.point1.x
                dy = line.point2.y - line.point1.y
                dz = line.point2.z - line.point1.z

                sum = abs(dx) + abs(dy) + abs(dz)

                # n in range (0, 1), for example if n1 is equal to 1, the length will only matter for the X axis
                n1 = dx / sum
                n2 = dy / sum
                n3 = dz / sum

                for n in range(0, int(line.len / size) - 1):
                    self.current_node = self.current_node + 1

                    #  coordinates of new node - based of calculated axis matter parameter - n
                    x = node1.x + size * n1
                    y = node1.y + size * n2
                    z = node1.z + size * n3

                    # creation new node and element based on two last created nodes
                    node2 = Node(x=x, y=y, z=z, index=self.current_node)
                    self.nodes.append(node2)
                    element = Element(node1, node2, line.material, line.section, line.direction_vector)

                    line.elements_index.append(len(self.elements))
                    self.lines.append(line)
                    self.elements.append(element)
                    element.node1.elements_index.append(len(self.elements) - 1)
                    element.node2.elements_index.append(len(self.elements) - 1)

                    node1 = node2

            if line.point2.node_number is None:
                # last node creation, at the end of the line - coordinates of node same as last point of line
                self.current_node = self.current_node + 1
                line.point2.node_number = self.current_node
                self.lines.append(line)
                node2 = Node(x=line.point2.x, y=line.point2.y, z=line.point2.z, index=self.current_node)
                self.nodes.append(node2)
                self.points.append(line.point2)
            else:
                # Condition if last node was already created on another line
                node2 = self.nodes[line.point2.node_number - 1]

            # creation last element on the line
            element = Element(node1, node2, line.material, line.section, line.direction_vector)
            line.elements_index.append(len(self.elements))
            self.lines.append(line)
            self.elements.append(element)
            element.node1.elements_index.append(len(self.elements) - 1)
            element.node2.elements_index.append(len(self.elements) - 1)

    def mesh_info(self) -> dict:
        if len(self.elements) > 0:
            minl = self.elements[0].L
            maxl = self.elements[0].L
            suml = 0

            for element in self.elements:
                length = element.L

                suml += length

                if minl > length:
                    minl = length
                if maxl < length:
                    maxl = length

            mean = suml / len(self.elements)

            return {'Quantity of elements': len(self.elements), 'Quantity of nodes': len(self.nodes),
                    'Average element size': mean, 'Max element size': maxl, 'Min element size': minl}
        else:
            return {}
