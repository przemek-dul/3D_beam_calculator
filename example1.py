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
point2 = Point(x=3, y=0, z=0, index=2)
point3 = Point(x=6, y=0, z=0, index=3)
point4 = Point(x=1.5, y=3, z=0, index=4)
point5 = Point(x=4.5, y=3, z=0, index=5)

point11 = Point(x=0, y=0, z=3, index=6)
point22 = Point(x=3, y=0, z=3, index=7)
point33 = Point(x=6, y=0, z=3, index=8)
point44 = Point(x=1.5, y=3, z=3, index=9)
point55 = Point(x=4.5, y=3, z=3, index=10)

line1 = Line(point1, point2, material, section)
line2 = Line(point2, point3, material, section)
line3 = Line(point1, point4, material, section)
line4 = Line(point4, point2, material, section)
line5 = Line(point4, point5, material, section)
line6 = Line(point2, point5, material, section)
line7 = Line(point5, point3, material, section)

line11 = Line(point11, point22, material, section)
line22 = Line(point22, point33, material, section)
line33 = Line(point11, point44, material, section)
line44 = Line(point44, point22, material, section)
line55 = Line(point44, point55, material, section)
line66 = Line(point22, point55, material, section)
line77 = Line(point55, point33, material, section)

line8 = Line(point1, point11, material, section)
line9 = Line(point2, point22, material, section)
line10 = Line(point3, point33, material, section)
line_11 = Line(point4, point44, material, section)
line_12 = Line(point5, point55, material, section)


load1 = Displacement(point1, DOF=True)
load2 = Displacement(point3, DOF=True)
load3 = Displacement(point11, DOF=True)
load4 = Displacement(point33, DOF=True)

load5 = Force(point5, 'y', 10000)
load6 = Pressure(line_11, 1, -100)

model = Model([line1, line2, line3, line4, line5, line6, line7, line11, line22, line33, line44, line55, line66, line77,
               line8, line9, line10, line_11, line_12
               ], displacement_bc=[load1, load2, load3, load4], forces_bc=[load5, load6])
model.mesh(3)
model.solve()

results = Results(model)
fig2 = results.nodal_deformation_3d(option='total', show_nodes=True, scale=1)

results.evaluate_all_results()
