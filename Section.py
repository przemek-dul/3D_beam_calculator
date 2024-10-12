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
        self.max_y = max_y  # the maxima distance from the edge to the center of the cross-section in y-direction
        self.max_z = max_z  # the maxima distance from the edge to the center of the cross-section in z-direction

        # returns shear correction factor depending on the shape of the cross-section
        self._shear_factor_fun = shear_factor_fun
        self.custom = True
        self.circular = False
        self.resolution = 300

        self.origin_point = [0, 0]
        self._current_section = ''
        self._dimensions = None

        self._z_points = None
        self._y_points = None

        self._mask = None

        self._check_input()

    def _shape_input(self, *args):
        self.custom = False
        for value in args:
            float_greater_than_0(value, 'dimensions of section must be FLOAT or INT',
                                 'dimensions of section must be greater than 0')

    def _bending_shear(self, values, option):
        S = np.empty(np.shape(values))
        width = np.empty(np.shape(values))

        if self._current_section == "rectangle":
            a_y, b_z = self._dimensions
            if option == 'z':
                S = (b_z / 2) * (a_y ** 2 / 4 - values ** 2)
                width = b_z
            elif option == 'y':
                S = (a_y / 2) * (b_z ** 2 / 4 - values ** 2)
                width = a_y

        elif self._current_section == "box":
            a_out, b_out, a_in, b_in = self._dimensions
            if option == 'z':
                for i in range(0, len(values[0])):
                    val = values[i][0]
                    if abs(val) < 0.5*a_in:
                        S[i, :] = (b_out / 2) * (a_out ** 2 / 4 - val ** 2) - (b_in / 2) * (a_in ** 2 / 4 - val ** 2)
                        width[i, :] = b_out - b_in
                    else:
                        S[i, :] = (b_out / 2) * (a_out ** 2 / 4 - val ** 2)
                        width[i, :] = b_out
            elif option == 'y':
                for i in range(0, len(values[0])):
                    val = values[0][i]
                    if abs(val) < 0.5*b_in:
                        S[:, i] = (a_out / 2) * (b_out ** 2 / 4 - val ** 2) - (a_in / 2) * (b_in ** 2 / 4 - val ** 2)
                        width[:, i] = a_out - a_in
                    else:
                        S[:, i] = (a_out / 2) * (b_out ** 2 / 4 - val ** 2)
                        width[:, i] = a_out

        elif self._current_section == "circle":
            radius = self._dimensions
            S = (2 / 3) * (radius ** 2 - values ** 2) ** 2
            width = np.sqrt(radius ** 2 - values ** 2)

        elif self._current_section == 'pipe':
            out_radius, in_radius = self._dimensions
            if option == 'z':
                for i in range(0, len(values[0])):
                    val = values[i][0]
                    if abs(val) < in_radius:
                        S[i, :] = (2 / 3) * ((out_radius ** 2 - val ** 2) ** 2 - (in_radius ** 2 - val ** 2) ** 2)
                        width[i, :] = np.sqrt(out_radius ** 2 - val ** 2) - np.sqrt(in_radius ** 2 - val ** 2)
                    else:
                        S[i, :] = (2 / 3) * ((out_radius ** 2 - val ** 2) ** 2)
                        width[i, :] = np.sqrt(out_radius ** 2 - val ** 2)
            elif option == 'y':
                for i in range(0, len(values[0])):
                    val = values[0][i]
                    if abs(val) < in_radius:
                        S[:, i] = (2 / 3) * ((out_radius ** 2 - val ** 2) ** 2 - (in_radius ** 2 - val ** 2) ** 2)
                        width[:, i] = np.sqrt(out_radius ** 2 - val ** 2) - np.sqrt(in_radius ** 2 - val ** 2)
                    else:
                        S[:, i] = (2 / 3) * ((out_radius ** 2 - val ** 2) ** 2)
                        width[:, i] = np.sqrt(out_radius ** 2 - val ** 2)

        elif self._current_section == "I_shape":
            a_y, b_z, t1, t2 = self._dimensions
            if option == 'z':
                for i in range(0, len(values[0])):
                    val = values[i][0]
                    if abs(val) < 0.5*(a_y - 2*t1):
                        S[i, :] = (t2 / 2) * (a_y ** 2 / 4 - val ** 2)
                        width[i, :] = t2
                    else:
                        S[i, :] = (b_z / 2) * (a_y ** 2 / 4 - val ** 2)
                        width[i, :] = b_z
            elif option == 'y':
                for i in range(0, len(values[0])):
                    val = values[0][i]
                    if abs(val) < 0.5*t2:
                        S[:, i] = (a_y / 2) * (b_z ** 2 / 4 - val ** 2)
                        width[:, i] = a_y
                    else:
                        S[:, i] = S[:, i] = (a_y / 2) * (b_z ** 2 / 4 - val ** 2) - ((a_y - 2 * t1) / 2) * (
                                b_z ** 2 / 4 - val ** 2)
                        width[:, i] = a_y - 2 * t1

        return S / width

    def _torsion_shear(self, T, z, y):
        output = np.ones(np.shape(z))
        if self._current_section == "circle" or self._current_section == 'pipe':
            J = self.Iz + self.Iy

            return np.sqrt(z ** 2 + y ** 2) * T / J

        else:
            return (T * max(self.max_z, self.max_y) / (self.Iz + self.Iy)) * output

    def _check_input(self):
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

        if self._shear_factor_fun is None:
            self._shear_factor_fun = lambda v: 5 / 6
        elif not callable(self._shear_factor_fun):
            raise TypeError("argument - shear_factor_fun must be lambda function")
        else:
            num_args = len(inspect.signature(self._shear_factor_fun).parameters)
            if num_args != 1:
                raise AttributeError("argument - shear_factor_fun must be lambda function of one input argument")

    #  basic shapes properties functions
    def rectangle(self, a_y: float, b_z: float):
        """
                     ▲  Y
                     │
                     │
                     │
                     │
                     │
     ──────▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
     ▲     ▌         │         ▐
     │     ▌         │         ▐
     │     ▌         │         ▐
     │     ▌         │         ▐     Z
a_y  │─────▌─────────•─────────▐─────►
     │     ▌         │         ▐
     │     ▌         │         ▐
     │     ▌         │         ▐
     ▼     ▌         │         ▐
     ──────▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
           │         │         │
           │         │         │
           │         │         │
           │◄─────────────────►│
                    b_z

        """

        self._shape_input(a_y, b_z)
        self._dimensions = [a_y, b_z]
        self._current_section = "rectangle"

        self.A = a_y * b_z
        self.Iy = a_y * b_z ** 3 / 12
        self.Iz = b_z * a_y ** 3 / 12
        self.max_z = 0.5 * b_z
        self.max_y = 0.5 * a_y

        self._shear_factor_fun = lambda v: 10 * (1 + v) / (12 + 11 * v)

        zz = np.linspace(-0.5 * b_z, 0.5 * b_z, self.resolution)
        yy = np.linspace(-0.5 * a_y, 0.5 * a_y, self.resolution)
        self._z_points, self._y_points = np.meshgrid(zz, yy)
        self._mask = lambda values: values

    def box(self, a_out: float, b_out: float, a_in: float, b_in: float):
        """
                             ▲  Y
                             │
                             │
                             │
                             │
                             │
                   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄──────────
                   ▌         │         ▐         ▲
              ─────▌─▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ▐         │
              ▲    ▌ ▌       │       ▐ ▐         │
              │    ▌ ▌       │       ▐ ▐     Z   │
        a_in  │────▌─▌───────•───────▐─▐─────►   │ a_out
              │    ▌ ▌       │       ▐ ▐         │
              ▼    ▌ ▌       │       ▐ ▐         │
              ─────▌ ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ ▐         │
                   ▌ │       │       │ ▐         ▼
                   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀──────────
                   │ │       │       │ │
                   │ │       │       │ │
                   │ │       │       │ │
                   │ │◄─────────────►│ │
                   │        b_in       │
                   │                   │
                   │◄─────────────────►│
                            b_out
        """

        # check input
        if a_in < a_out or b_in < b_out:
            raise AttributeError("Cross-sections geometry defined improperly. Check input dimensions.")

        self._shape_input(a_out, b_out, a_in, b_in)
        self._dimensions = [a_out, b_out, a_in, b_in]
        self._current_section = "box"

        self.A = a_out * b_out - a_in * b_in
        self.Iy = a_out * b_out ** 3 / 12 - a_in * b_in ** 3 / 12
        self.Iz = b_out * a_out ** 3 / 12 - b_in * a_in ** 3 / 12
        self.max_z = 0.5 * b_out
        self.max_y = 0.5 * a_out

        t1 = (a_out - a_in) / 2
        t2 = (b_out - b_in) / 2
        m = b_out * t1 / a_out * t2
        n = b_out / a_out

        self._shear_factor_fun = lambda v: 10 * (1 + v) * (1 + 3 * m) ** 2 / (
                    (12 + 72 * m + 150 * m ** 2 + 90 * m ** 3) + v * (
                        11 + 66 * m + 135 * m ** 2 + 90 * m ** 3) + 10 * n ** 2 * ((3 + v) * m + 3 * m ** 2))

        zz = np.linspace(-0.5 * b_out, 0.5 * b_out, self.resolution)
        yy = np.linspace(-0.5 * a_out, 0.5 * a_out, self.resolution)
        self._z_points, self._y_points = np.meshgrid(zz, yy)
        self._mask = lambda values: np.ma.masked_where(
            np.logical_and(self._z_points ** 2 < (0.5 * b_in) ** 2, self._y_points ** 2 < (0.5 * a_in) ** 2), values)

    def circle(self, diameter: float):
        """
                     ▲  Y
                     │
                     │
                     │
                     │
               ▀▀▀▀▀▀▀▀▀▀▀▀▀
              ▀▀     │     ▀▀
             ▀▀      │      ▀▀
            ▀▀       │       ▀▀
           ▀▀        │        ▀▀
          ▀▀         │         ▀▀    Z
        ──▌ ─────────•───────── ▐────►
          ▄▄         │         ▄▄
          │▄▄        │        ▄▄│
          │ ▄▄       │       ▄▄ │
          │  ▄▄      │      ▄▄  │
          │   ▄▄     │     ▄▄   │
          │     ▄▄▄▄▄▄▄▄▄▄▄▄    │
          │                     │
          │          │          │
          │◄───────────────────►│
                 diameter
        """

        self._shape_input(diameter)
        self._dimensions = diameter / 2
        self._current_section = "circle"
        self.circular = True

        radius = diameter / 2
        self.A = np.pi * radius ** 2
        self.Iz = np.pi * radius ** 4 / 4
        self.Iy = np.pi * radius ** 4 / 4
        self.max_z = radius
        self.max_y = radius

        self._shear_factor_fun = lambda v: 6 * (1 + v) / (7 + 6 * v)

        zz = np.linspace(-0.999 * 0.5 * diameter, 0, int(self.resolution / 2))
        zz = np.append(zz, np.linspace(-0, 0.999 * 0.5 * diameter, int(self.resolution / 2)))
        yy = zz

        self._z_points, self._y_points = np.meshgrid(zz, yy)
        self._mask = lambda values: np.ma.masked_where(self._z_points ** 2 + self._y_points ** 2 > radius ** 2, values)

    def pipe(self, out_diameter: float, in_diameter: float):
        """
                     ▲  Y
                     │
                     │
                     │
                     │
               ▀▀▀▀▀▀▀▀▀▀▀▀▀
              ▀▀     │     ▀▀
             ▀▀  ▀▀▀▀▀▀▀▀▀  ▀▀
            ▀▀  ▀▀   │   ▀▀  ▀▀
           ▀▀  ▀▀    │    ▀▀  ▀▀
          ▀▀  ▀▀     │     ▀▀  ▀▀    Z
        ──▌ ──▌──────•──────▐── ▐────►
          ▄▄  ▄▄     │     ▄▄  ▄▄
          │▄▄ │▄▄    │    ▄▄│ ▄▄│
          │ ▄▄│ ▄▄   │   ▄▄ │▄▄ │
          │  ▄▄  ▄▄▄▄▄▄▄▄▄  ▄▄  │
          │   ▄▄     │     ▄▄   │
          │   │▄▄▄▄▄▄▄▄▄▄▄▄▄│   │
          │   │             │   │
          │   │      │      │   │
          │   │◄───────────►│   │
          │     in_diameter     │
          │                     │
          │◄───────────────────►│
                out_diameter
        """
        # check input
        if out_diameter < in_diameter:
            raise AttributeError("Cross-sections geometry defined improperly. Check input dimensions.")

        self._shape_input(in_diameter, out_diameter)
        in_radius = in_diameter / 2
        out_radius = out_diameter / 2
        self._dimensions = [out_radius, in_radius]
        self._current_section = "pipe"
        self.circular = True

        self.A = np.pi * out_radius ** 2 - np.pi * in_radius ** 2
        self.Iz = np.pi * out_radius ** 4 / 4 - np.pi * in_radius ** 4 / 4
        self.Iy = np.pi * out_radius ** 4 / 4 - np.pi * in_radius ** 4 / 4
        self.max_z = out_radius
        self.max_y = out_radius

        m = in_radius / out_radius

        self._shear_factor_fun = lambda v: 6 * (1 + v) * (1 + m ** 2) ** 2 / (
                (7 + 6 * v) * (1 + m ** 2) ** 2 + (20 + 12 * v) * m ** 2)

        zz = np.linspace(-0.999 * 0.5 * out_diameter, 0.999 * 0.5 * out_diameter, self.resolution)
        yy = np.linspace(-0.999 * 0.5 * out_diameter, 0.999 * 0.5 * out_diameter, self.resolution)
        self._z_points, self._y_points = np.meshgrid(zz, yy)
        self._mask = lambda values: np.ma.masked_where(
            (self._z_points ** 2 + self._y_points ** 2 > out_radius ** 2) |
            (self._z_points ** 2 + self._y_points ** 2 < in_radius ** 2),
            values)

    def I_shape(self, a_y: float, b_z: float, t1: float, t2: float):
        """
                               ▲  Y
                               │
                               │
                         t2    │
                       ──────◄───►
                             │ │ │
                       ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄────────────
                       ▌     │ │ │     ▐           ▲
                       ▀▀▀▀▀▀▌   ▐▀▀▀▀▀▀           │
                             ▌ │ ▐                 │
                             ▌ │ ▐             Z   │
                  │     ─────▌─•─▐─────────────►   │ a_y
               t1 │          ▌ │ ▐                 │
                  │          ▌ │ ▐                 │
                  ▲────▄▄▄▄▄▄▌   ▐▄▄▄▄▄▄           │
                  │    ▌       │       ▐           ▼
                  ▼────▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀────────────
                       │       │       │
                       │       │       │
                       │       │       │
                       │◄─────────────►│
                              b_z
                """

        if a_y < 2*t1 or b_z < t2:
            raise AttributeError("Cross-sections geometry defined improperly. Check input dimensions.")

        self._shape_input(a_y, b_z, t1, t2)
        self._dimensions = [a_y, b_z, t1, t2]
        self._current_section = "I_shape"

        self.A = a_y * b_z - 2 * (a_y - 2 * t1) * 0.5 * (b_z - t2)
        self.Iy = (t2**3*(a_y - 2 * t1) / 12) + (b_z**3 / 12)*(a_y - (a_y - 2 * t1))
        self.Iz = (t2 * (a_y - 2 * t1) ** 3 / 12) + ((a_y - 2 * t1) / 12) * (a_y ** 3 - (a_y - 2 * t1) ** 3)
        self.max_z = 0.5 * b_z
        self.max_y = 0.5 * a_y

        m = 2 * b_z * t1 / a_y * t2
        n = b_z / a_y

        self._shear_factor_fun = lambda v: 10 * (1 + v) * (1 + 3 * m) ** 2 / (
                (12 + 72 * m + 150 * m ** 2 + 90 * m ** 3) + v * (
                11 + 66 * m + 135 * m ** 2 + 90 * m ** 3) + 30 * n ** 2 * (m + m ** 2) + 5 * v * n ** 2 * (
                            8 * m + 9 * m ** 2))

        zz = np.linspace(-0.5 * b_z, 0.5 * b_z, self.resolution)
        yy = np.linspace(-0.5 * a_y, 0.5 * a_y, self.resolution)
        self._z_points, self._y_points = np.meshgrid(zz, yy)
        self._mask = lambda values: np.ma.masked_where(
            np.logical_and(self._z_points ** 2 > (0.5 * t2) ** 2, self._y_points ** 2 < (0.5 * (a_y - 2 * t1)) ** 2),
            values)
