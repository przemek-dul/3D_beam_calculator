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

# first rect section definition
a1 = 0.3
b1 = 0.5
section_1 = Section()
section_1.rectangle(a_y=a1, b_z=b1)

# second rect section definition
a2 = 0.5
b2 = 0.2
section_2 = Section()
section_2.rectangle(a_y=a2, b_z=b2)

# geometry definition
point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=0, y=3, z=0, index=2)
point3 = Point(x=3, y=3, z=0, index=3)
point4 = Point(x=3, y=0, z=0, index=4)

cross_section_orientation = [0, 0, -1]  # direction of z axis for cross-section on line1, line2 and line3

line1 = Line(point1, point2, material, section_1, cross_section_orientation)
line2 = Line(point2, point3, material, section_2, cross_section_orientation)
line3 = Line(point3, point4, material, section_1, cross_section_orientation)

# mesh definition - elements of length 0.1 for every line
mesh = Mesh()
mesh.max_element_size(size=0.1, lines=[line1, line2, line3])

# boundary conditions definition
displacement1 = Displacement(point1, uy=0, uz=0, rot_x=0, rot_y=0)
displacement2 = Displacement(point4, ux=0, uy=0, uz=0, rot_x=0, rot_y=0)

pressure1 = Pressure(line=line1, direction='y', value=-1000)
torque1 = Torque(point=point3, axis='z', value=-9000)

# model definition and solve
model = Static(mesh, displacement_bc=[displacement1, displacement2], forces_bc=[pressure1, torque1])
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='uy', show_undeformed=True, show_points=True)
fig2 = results.deformation_3d(option='ux', show_undeformed=True, show_points=True)
fig3 = results.deformation_3d(option='rotz', show_points=True, show_undeformed=True)

fig4 = results.bar_force_3d(option='fy')
fig5 = results.bar_force_3d(option='mz')

# save graphs
#  pl.offline.plot(fig1, filename='deformations-y_direction.html', auto_open=False)
#  pl.offline.plot(fig2, filename='deformations-x_direction.html', auto_open=False)
#  pl.offline.plot(fig3, filename='rotation-z_axis.html', auto_open=False)
#  pl.offline.plot(fig4, filename='shear_force-y_direction.html', auto_open=False)
#  pl.offline.plot(fig5, filename='bending_moment-z_axis.html', auto_open=False)

results.evaluate_all_results()