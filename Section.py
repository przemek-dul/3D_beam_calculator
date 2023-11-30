import numpy as np


class Section:
    def __init__(self, A=0, Iz=0, Iy=0, Qz=0, Qy=0, max_z=0, max_y=0, z_wth=0, y_wth=0):
        self.A = A
        self.Iz = Iz
        self.Iy = Iy
        self.Qz = Qz
        self.Qy = Qy
        self.max_y = max_y
        self.max_z = max_z
        self.z_wth = z_wth
        self.y_wth = y_wth

    def rectangle(self, a_y, b_z):
        self.A = a_y * b_z
        self.Iz = a_y * b_z**3 / 12
        self.Iy = b_z * a_y ** 3 / 12
        self.Qz = a_y * b_z ** 2 / 12
        self.Qy = b_z * a_y ** 2 / 12
        self.max_z = 0.5 * b_z
        self.max_y = 0.5 * a_y
        self.z_wth = b_z
        self.y_wth = a_y

    def circle(self, diameter):
        radius = diameter / 2
        self.A = np.pi * radius ** 2
        self.Iz = np.pi * radius ** 4 / 4
        self.Iy = np.pi * radius ** 4 / 4
        self.Qz = np.pi * radius ** 3 / 4
        self.Qy = np.pi * radius ** 3 / 4
        self.max_z = radius
        self.max_y = radius
        self.z_wth = diameter
        self.y_wth = diameter

    def rectangle_in_rectangle(self, a_y, b_z, a_inner, b_inner):
        self.A = a_y * b_z - a_inner * b_inner
        self.Iz = (a_y * b_z ** 3 - a_inner * b_inner ** 3) / 12
        self.Iy = (b_z * a_y ** 3 - b_inner * a_inner ** 3) / 12
        self.Qz = (a_y * b_z ** 2 - a_inner * b_inner ** 2) / 12
        self.Qy = (b_z * a_y ** 2 - b_inner * a_inner ** 2) / 12
        self.max_z = 0.5 * b_z
        self.max_y = 0.5 * a_y
        self.z_wth = b_z - b_inner
        self.y_wth = a_y - a_inner
