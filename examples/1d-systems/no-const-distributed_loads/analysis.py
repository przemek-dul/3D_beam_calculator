from Section import Section
from Mesh import Mesh
from Line import Line
from Point import Point
from Model import Static
from Load import Displacement, Force, Torque, Pressure
from Material import Material
from Static_results import Static_results as Results
import plotly as pl

"""
In Ansys APDL software there is no opportunity to define non-constant distributed load for more than one  beam element,
so in this case there is no comparison of analysis results.
"""

# material definition
E = 200000000000
v = 0.3
material = Material(E=E, v=v)

# pipe section definition
outer_diameter = 0.25
inner_diameter = 0.18

section = Section()
section.pipe(out_diameter=outer_diameter, in_diameter=inner_diameter)

# geometry definition
point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=2, y=0, z=0, index=2)
point3 = Point(x=4, y=0, z=0, index=3)

line1 = Line(point1=point1, point2=point2, material=material, section=section)
line2 = Line(point1=point2, point2=point3, material=material, section=section)

# mesh definition - 50 elements per line
mesh = Mesh()
mesh.elements_on_line(lines=[line1, line2], num=50)

# boundary conditions definition

# 1-st non_const pressure vector (triangle)
v1 = [-4000, 0]

#  2-nd non_const pressure vector (triangle)
v2 = [2500, 1250]

#  3-rd non_const pressure vector (triangle)
v3 = [1250, 0]

pressure1 = Pressure(line=line1, direction='y', value=v1)
pressure2 = Pressure(line=line1, direction='y', value=v2)
pressure3 = Pressure(line=line2, direction='y', value=v3)

displacement1 = Displacement(point=point3, DOF=True)

# model definition and solve
model = Static(mesh=mesh, displacement_bc=[displacement1], forces_bc=[pressure1, pressure2, pressure3])
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='uy', show_undeformed=True, show_points=True)
fig2 = results.deformation_3d(option='rotz', show_points=True, show_undeformed=True)

fig3 = results.bar_force_3d(option='fy')
fig4 = results.bar_force_3d(option='mz')

# save graphs
# pl.offline.plot(fig1, filename='deformations-y_direction.html', auto_open=False)
# pl.offline.plot(fig2, filename='rotation-z_axis.html', auto_open=False)
# pl.offline.plot(fig3, filename='shear_force-y_direction.html', auto_open=False)
# pl.offline.plot(fig4, filename='bending_moment-z_axis.html', auto_open=False)

results.evaluate_all_results()
