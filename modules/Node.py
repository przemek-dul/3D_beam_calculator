import numpy as np


class Node:
    def __init__(self, x, y, z, index):
        self.x = x
        self.y = y
        self.z = z

        self.index = index
        self.elements_index = []  # indexes of elements built by node
        self.points_vector = np.array([x, y, z])
        self.displacement_vector = np.empty((1, 6))




