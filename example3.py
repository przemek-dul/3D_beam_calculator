from Material import Material
from Point import Point
from Section import Section
from Line import Line
from Model import Model
from Load import Force, Displacement, Pressure, Torque
from Results import Results


material = Material(E=200000000000, v=0.3)
section = Section()
section.rectangle(0.5, 0.1)

point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=2, y=0, z=0, index=2)
point3 = Point(x=5, y=0, z=0, index=1)


line1 = Line(point1, point2, material, section)
line2 = Line(point2, point3, material, section)


load_d1 = Displacement(point1, ux=0, uy=0, uz=0, fx=0, fy=0)
load_d2 = Displacement(point3, uy=0, uz=0, fx=0, fy=0)

input1 = Torque(point2, 'z', -1000)

model = Model([line1, line2], displacement_bc=[load_d1, load_d2], forces_bc=[input1])
model.mesh(25)
model.solve()

results = Results(model)

fig11 = results.nodal_deformation_3d('ux')
#fig12 = results.nodal_deformation('fz')

#fig2 = results.force_torque_plot('My')
#fig3 = results.force_torque_plot('Fy')

results.evaluate_all_results()