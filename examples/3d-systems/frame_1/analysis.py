from Section import Section
from Mesh import Mesh
from Line import Line
from Point import Point
from Model import Static
from Load import Displacement, Force, Torque, Pressure
from Material import Material
from Static_results import Static_results as Results
import plotly as pl

# material definition
E = 200000000000
v = 0.3
material = Material(E=E, v=v)

# circle section definition
d = 0.15
section = Section()
section.circle(diameter=d)

# geometry definition
point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=0, y=1, z=0, index=2)
point3 = Point(x=2, y=1, z=0, index=3)

point4 = Point(x=0, y=0, z=2, index=4)
point5 = Point(x=0, y=1, z=2, index=5)
point6 = Point(x=2, y=1, z=2, index=6)


line1 = Line(point1, point3, material, section, [0, 0, -1])
line2 = Line(point2, point3, material, section, [0, 0, -1])

line3 = Line(point3, point6, material, section, [0, -1, 0])

line4 = Line(point4, point6, material, section, [0, 0, 1])
line5 = Line(point5, point6, material, section, [0, 0, 1])

# mesh definition - elements of length 0.1 for every line
mesh = Mesh()
mesh.max_element_size(size=0.1, lines=[line1, line2, line3, line4, line5])

# boundary conditions definition
displacement1 = Displacement(point1, DOF=True)
displacement2 = Displacement(point2, DOF=True)
displacement3 = Displacement(point4, DOF=True)
displacement4 = Displacement(point5, DOF=True)

pressure1 = Pressure(line=line3, direction='z', value=12000)

# model definition and solve
model = Static(mesh, displacement_bc=[displacement1, displacement2, displacement3, displacement4], forces_bc=[pressure1])
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='total_disp', show_undeformed=True, show_points=True)
fig2 = results.deformation_3d(option='total_rot', show_points=True, show_undeformed=True)
fig3 = results.stress_3d(option='total', show_points=True, show_undeformed=True)

# save graphs
#  pl.offline.plot(fig1, filename='total_deformation.html', auto_open=False)
#  pl.offline.plot(fig2, filename='total_rotation.html', auto_open=False)
#  pl.offline.plot(fig3, filename='von-mises-stress.html', auto_open=False)

results.evaluate_all_results()