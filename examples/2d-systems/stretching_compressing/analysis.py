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

# rect section definition
a = 0.2
b = 0.2
section = Section()
section.rectangle(a_y=a, b_z=b)

# geometry definition
point1 = Point(x=0, y=2, z=0, index=1)
point2 = Point(x=2, y=2, z=0, index=2)
point3 = Point(x=4, y=2, z=0, index=3)
point4 = Point(x=2, y=0, z=0, index=4)

cross_section_orientation = [0, 0, -1]  # direction of z axis for cross-section on line1, line2 and line3

line1 = Line(point1, point4, material, section, cross_section_orientation)
line2 = Line(point2, point4, material, section, cross_section_orientation)
line3 = Line(point3, point4, material, section, cross_section_orientation)

# mesh definition - elements of length 0.05 for every line
mesh = Mesh()
mesh.max_element_size(size=0.05, lines=[line1, line2, line3])

# boundary conditions definition
displacement1 = Displacement(point1, ux=0, uy=0, uz=0, rot_x=0, rot_y=0)
displacement2 = Displacement(point2, ux=0, uy=0, uz=0, rot_x=0, rot_y=0)
displacement3 = Displacement(point3, ux=0, uy=0, uz=0, rot_x=0, rot_y=0)

force = Force(point=point4, direction='y', value=-20000)

# model definition and solve
model = Static(mesh, displacement_bc=[displacement1, displacement2, displacement3], forces_bc=[force])
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='uy', show_undeformed=True)
fig2 = results.stress_3d('nx')

#  show residuals forces
residuals = results.residuals_at_bc_points()
print(residuals)

# save graphs
#  pl.offline.plot(fig1, filename='deformations-y_direction.html', auto_open=False)
#  pl.offline.plot(fig2, filename='normal_stress_due_to_stretch.html', auto_open=False)

results.evaluate_all_results()