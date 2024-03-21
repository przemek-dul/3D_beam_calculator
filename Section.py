import numpy as np
import inspect


def float_greater_than_0(value, text1, text2):
    if value is not None:
        if type(value) != float and type(value) != int:
            raise TypeError(text1)
        elif value <= 0:
            raise ValueError(text2)


class Section:
    def __init__(self, A: float = None, Iz: float = None, Iy: float = None, max_z: float = None, max_y: float = None,
                 shear_factor_fun=None):
        self.A = A  # area of section
        self.Iz = Iz  # second moment of inertia about the Z axis
        self.Iy = Iy  # second moment of inertia about the Y axis
        # self.Qz = Qz  # first moment of inertia about the Z axis
        # self.Qy = Qy  # first moment of inertia about the Y axis
        self.max_y = max_y  # the maxima distance from the edge to the center of the cross-section in y-direction
        self.max_z = max_z  # the maxima distance from the edge to the center of the cross-section in z-direction
        # self.z_wth = z_wth  # width of cross-section in z-direction
        # self.y_wth = y_wth  # width of cross-section in y-direction

        # returns shear correction factor depending on the shape of the cross-section
        self.shear_factor_fun = shear_factor_fun

        self.check_input()

    def shape_input(self, *args):
        for value in args:
            float_greater_than_0(value, 'dimensions of section must be FLOAT or INT',
                                 'dimensions of section must be greater than 0')

    def check_input(self):
        float_greater_than_0(self.A, "argument - A (area) must be FLOAT or INT",
                             "argument - A (area) - must be greater than 0")

        float_greater_than_0(self.Iz, "argument - Iz (second moment of inertia) must be FLOAT or INT",
                             "argument - Iz (second moment of inertia) must be greater than 0")

        float_greater_than_0(self.Iy, "argument - Iy (second moment of inertia) must be FLOAT or INT",
                             "argument - Iy (second moment of inertia) must be greater than 0")

        float_greater_than_0(self.max_z, "argument - max_z (max distance from z axis) must be FLOAT or INT",
                             "argument - max_z (max distance from z axis) must be greater than 0")

        float_greater_than_0(self.max_y, "argument - max_y (max distance from y axis) must be FLOAT or INT",
                             "argument - max_y (max distance from y axis) must be greater than 0")

        if self.shear_factor_fun is None:
            self.shear_factor_fun = lambda v: 5 / 6
        elif not callable(self.shear_factor_fun):
            raise TypeError("argument - shear_factor_fun must be lambda function")
        else:
            num_args = len(inspect.signature(self.shear_factor_fun).parameters)
            if num_args != 1:
                raise AttributeError("argument - shear_factor_fun must be lambda function of ine input argument")

    #  basic shapes properties functions
    def rectangle(self, a_y: float, b_z: float):
        self.shape_input(a_y, b_z)

        self.A = a_y * b_z
        self.Iz = a_y * b_z ** 3 / 12
        self.Iy = b_z * a_y ** 3 / 12
        self.max_z = 0.5 * b_z
        self.max_y = 0.5 * a_y

        self.shear_factor_fun = lambda v: 10 * (1 + v) / (12 + 11 * v)

    def circle(self, diameter: float):
        self.shape_input(diameter)

        radius = diameter / 2
        self.A = np.pi * radius ** 2
        self.Iz = np.pi * radius ** 4 / 4
        self.Iy = np.pi * radius ** 4 / 4
        self.max_z = radius
        self.max_y = radius

        self.shear_factor_fun = lambda v: 6 * (1 + v) / (7 + 6 * v)

    def pipe(self, out_diameter: float, in_diameter: float):
        in_radius = in_diameter / 2
        out_radius = out_diameter / 2
        self.A = np.pi * out_radius ** 2 - np.pi * in_radius ** 2
        self.Iz = np.pi * out_radius ** 4 / 4 - np.pi * in_radius ** 4 / 4
        self.Iy = np.pi * out_radius ** 4 / 4 - np.pi * in_radius ** 4 / 4
        self.max_z = out_radius
        self.max_y = out_radius

        m = in_radius / out_radius

        self.shear_factor_fun = lambda v: 6 * (1+v) * (1+m**2)**2 / ((7+6*v)*(1+m**2)**2+(20+12*v)*m**2)
