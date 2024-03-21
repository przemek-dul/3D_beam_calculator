class Material:
    def __init__(self, E: float, v: float):
        self.E = E  # Young's module
        self.v = v  # Poisson's ratio

        self._check_input()

    def _check_input(self):
        if type(self.E) != float and type(self.E) != int:
            raise TypeError("argument - E (Young's module) must be FLOAT or INT")
        elif self.E <= 0:
            raise ValueError("argument - E (Young's module) must be greater than 0")

        if type(self.v) != float and type(self.v) != int:
            raise TypeError("argument - v (Poisson's ratio) must be FLOAT or INT")
        elif self.v <= 0:
            raise ValueError("argument - v (Poisson's ratio) must be greater than 0")

