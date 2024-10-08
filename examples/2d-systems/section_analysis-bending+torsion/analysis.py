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

#  circle section definition
diameter = 0.02

section1 = Section()
section1.circle(diameter=diameter)


# geometry definition
point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=1, y=0, z=0, index=2)
point3 = Point(x=1, y=0, z=0.05, index=3)

# direction vector for line2 (by default line 2 is parallel to default direction vector)
v1 = [-1, 0, 0]

line1 = Line(point1=point1, point2=point2, material=material, section=section1, index=1)
line2 = Line(point1=point2, point2=point3, material=material, section=section1, direction_vector=v1, index=2)


# mesh definition - 25 elements per line
mesh = Mesh()
mesh.elements_on_line(lines=[line1, line2], num=25)

# boundary conditions definition
force1 = Force(point=point3, direction='y', value=-1000)

displacement1 = Displacement(point=point1, DOF=True)

# model definition and solve
model = Static(mesh=mesh, displacement_bc=[displacement1], forces_bc=[force1], analytical_shear_stresses=True)
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d(option='uy', show_undeformed=True)
fig2 = results.deformation_3d(option='rotz', show_undeformed=True)

fig3 = results.stress_3d('total')
fig4 = results.stress_3d('st')

#  section stress distribution
fig5 = results.section_stress('ny', line1, 0)
fig6 = results.section_stress('st', line1, 0)
fig7 = results.section_stress('total', line1, 0)

results.evaluate_all_results()