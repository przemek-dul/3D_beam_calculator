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
point2 = Point(x=5, y=0, z=0, index=2)
point3 = Point(x=0, y=0, z=5, index=3)
point4 = Point(x=5, y=0, z=5, index=4)

point5 = Point(x=1, y=3, z=0, index=5)
point6 = Point(x=4, y=3, z=0, index=6)
point7 = Point(x=1, y=3, z=5, index=7)
point8 = Point(x=4, y=3, z=5, index=8)

point9 = Point(x=1, y=9, z=0, index=9)
point10 = Point(x=4, y=9, z=0, index=10)
point11 = Point(x=1, y=9, z=5, index=11)
point12 = Point(x=4, y=9, z=5, index=12)

line1 = Line(point1, point2, material, section, [0, -1, 0])
line2 = Line(point2, point4, material, section, [0, -1, 0])
line3 = Line(point4, point3, material, section, [0, -1, 0])
line4 = Line(point3, point1, material, section, [0, -1, 0])

line5 = Line(point1, point5, material, section, [0, 0, -1])
line6 = Line(point2, point6, material, section, [0, 0, -1])
line7 = Line(point3, point7, material, section, [0, 0, -1])
line8 = Line(point4, point8, material, section, [0, 0, -1])

line9 = Line(point5, point6, material, section, [0, -1, 0])
line10 = Line(point6, point8, material, section, [0, -1, 0])
line11 = Line(point8, point7, material, section, [0, -1, 0])
line12 = Line(point7, point5, material, section, [0, -1, 0])

line13 = Line(point5, point9, material, section, [0, 0, -1])
line14 = Line(point6, point10, material, section, [0, 0, -1])
line15 = Line(point7, point11, material, section, [0, 0, -1])
line16 = Line(point8, point12, material, section, [0, 0, -1])

line17 = Line(point9, point10, material, section, [0, -1, 0])
line18 = Line(point10, point12, material, section, [0, -1, 0])
line19 = Line(point12, point11, material, section, [0, -1, 0])
line20 = Line(point11, point9, material, section, [0, -1, 0])

line21 = Line(point5, point10, material, section, [0, 0, -1])
line22 = Line(point6, point12, material, section, [0, -1, 0])
line23 = Line(point7, point12, material, section, [0, 0, -1])
line24 = Line(point7, point9, material, section, [0, -1, 0])


# mesh definition - elements of length 0.1 for every line
mesh = Mesh()
mesh.max_element_size(size=0.1, lines=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11,
                                       line12, line13, line14, line15, line16, line17, line18, line19, line20, line21,
                                       line22, line23, line24])

# boundary conditions definition
displacement1 = Displacement(point1, DOF=True)
displacement2 = Displacement(point2, DOF=True)
displacement3 = Displacement(point3, DOF=True)
displacement4 = Displacement(point4, DOF=True)

pressure1 = Pressure(line=line13, direction='y', value=-1200)
pressure2 = Pressure(line=line16, direction='y', value=1800)

force1 = Force(point=point9, direction='y', value=-120000)
force2 = Force(point=point12, direction='y', value=-150000)

# model definition and solve
model = Static(mesh, displacement_bc=[displacement1, displacement2, displacement3, displacement4],
               forces_bc=[pressure1, pressure2, force1, force2])
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='total_disp')
fig2 = results.deformation_3d(option='total_rot')
fig3 = results.stress_3d(option='total')

# save graphs
pl.offline.plot(fig1, filename='total_deformation.html', auto_open=False)
pl.offline.plot(fig2, filename='total_rotation.html', auto_open=False)
pl.offline.plot(fig3, filename='von-mises-stress.html', auto_open=False)

results.evaluate_all_results()