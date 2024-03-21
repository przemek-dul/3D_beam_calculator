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
E = 320000000000
v = 0.15
material = Material(E=E, v=v)

# first circle section definition
diameter = 0.1

section1 = Section()
section1.circle(diameter=diameter)

# second circle section definition
diameter = 0.04

section2 = Section()
section2.circle(diameter=diameter)

# geometry definition
point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=0, y=4, z=0, index=2)
point3 = Point(x=0, y=5, z=0, index=3)
point4 = Point(x=2, y=5, z=0, index=4)


line1 = Line(point1=point1, point2=point2, material=material, section=section1)
line2 = Line(point1=point2, point2=point3, material=material, section=section1)
line3 = Line(point1=point3, point2=point4, material=material, section=section2)
line4 = Line(point1=point2, point2=point4, material=material, section=section2)

# mesh definition - 100 elements per line
mesh = Mesh()
mesh.elements_on_line(lines=[line1, line2, line3, line4], num=100)

# boundary conditions definition
force1 = Force(point=point4, direction='y', value=-1800)

displacement1 = Displacement(point=point1, DOF=True)

# model definition and solve
model = Static(mesh=mesh, displacement_bc=[displacement1], forces_bc=[force1])
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='uy', show_undeformed=True)
fig2 = results.deformation_3d(option='ux', show_undeformed=True)
fig3 = results.deformation_3d(option='rotz', show_undeformed=True)

fig4 = results.stress_3d('total')

# save graphs
#  pl.offline.plot(fig1, filename='deformations-y_direction.html', auto_open=False)
#  pl.offline.plot(fig2, filename='deformations-x_direction.html', auto_open=False)
#  pl.offline.plot(fig3, filename='rotation-z_axis.html', auto_open=False)
#  pl.offline.plot(fig4, filename='von_Mises_stress.html', auto_open=False)

results.evaluate_all_results()