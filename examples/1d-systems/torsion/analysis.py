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
v = 0.2
material = Material(E=E, v=v)

# circle section definition
d1 = 0.02
section1 = Section()
section1.circle(diameter=d1)

# pipe section definition
d2 = 0.03
d3 = 0.01
section2 = Section()
section2.pipe(out_diameter=d2, in_diameter=d3)

# geometry definition
point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=0.4, y=0, z=0, index=2)
point3 = Point(x=0.8, y=0, z=0, index=3)

line1 = Line(point1, point2, material, section1)
line2 = Line(point2, point3, material, section2)

# mesh definition - 50 elements per line
mesh = Mesh()
mesh.elements_on_line(lines=[line1, line2], num=50)

# boundary conditions definition
torque1 = Torque(point=point2, axis='x', value=600)

displacement1 = Displacement(point=point1, DOF=True)
displacement2 = Displacement(point=point3, DOF=True)

# model definition and solve
model = Static(mesh=mesh, displacement_bc=[displacement1, displacement2], forces_bc=[torque1])
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='rotx', show_points=True)
fig2 = results.stress_3d(option='st')
fig3 = results.bar_force_3d(option='mx')

# save graphs
#  pl.offline.plot(fig1, filename='rotation-x_axis.html', auto_open=False)
#  pl.offline.plot(fig2, filename='max_shear_stress_due_to_torsion.html', auto_open=False)
#  pl.offline.plot(fig3, filename='torsion_moment.html', auto_open=False)

results.evaluate_all_results()