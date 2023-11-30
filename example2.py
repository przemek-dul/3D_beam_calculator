from Material import Material
from Point import Point
from Section import Section
from Line import Line
from Model import Model
from Load import Force, Displacement, Pressure, Torque
from Results import Results

q = 1000
a = 3

material = Material(E=200000000000, v=0.3)
section = Section()
section.rectangle(0.5, 0.1)

point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=0, y=a, z=0, index=2)
point3 = Point(x=a, y=a, z=0, index=3)
point4 = Point(x=a, y=0, z=0, index=4)

line1 = Line(point1, point2, material, section)
line2 = Line(point2, point3, material, section)
line3 = Line(point4, point3, material, section)


load_d1 = Displacement(point1, uy=0, uz=0, fx=0, fy=0)
load_d2 = Displacement(point4, ux=0, uy=0, uz=0, fx=0, fy=0)

load_pressure1 = Pressure(line1, 1, -1000)
load_torque1 = Torque(point3, 'z', 1000*9)


model = Model([line1, line2, line3], displacement_bc=[load_d1, load_d2], forces_bc=[load_pressure1, load_torque1])

model.mesh(50)
model.solve()

results = Results(model)

for node in model.nodes:
    print(node.uy)
fig2 = results.force_torque_plot('My')
fig3 = results.force_torque_plot('Fy')

results.evaluate_all_results()