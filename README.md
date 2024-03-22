3D Beam Calculator is CAE software used for simulating loads on static beam structures within the linear range, applying Hooke's law.
The software utilizes the Finite Element Method (FEM) to solve the differential equations governing beam deflection, implementing one
type of finite element based on the Timoshenko–Ehrenfest beam theory. The calculator computes deformations, stresses, and internal forces
within the structure. 

Documentation:
------------
Documentation will be created soon.

Required Libraries:
------------
- **Numpy:** https://www.numpy.org
- **Matplotlib:** https://www.matplotlib.org
- **Plotly:** https://www.plotly.com/python/
- **mplcursors:** https://www.mplcursors.readthedocs.io/en/stable/
- **Loguru:** https://loguru.readthedocs.io/en/stable/
- **PyGt5:** https://pypi.org/project/PyQt5/

All required libraries can be installed via pip using the following command:
::

   pip install "libname"

Installation
------------
Currently, the software can only be installed by cloning the repository. This can be achieved using Git with the following command:
::

   git clone https://github.com/przemek-dul/3D_beam_calculator

System requirements
------------
The software is compatible with Windows and Linux systems. Compatibility with iOS has not been tested. On Linux, the Plotly result graphs
may not display in the default library window and should be opened in a browser window instead. Python 3.10.8 or later is recommended.

Software development plans
------------
Currently, the static analysis is complete. In the future, there are plans to implement modal and harmonic analysis, improve the process
of creating system geometry, add more basic cross-sections, and perhaps introduce a new type of element based on the Euler–Bernoulli beam theory.




