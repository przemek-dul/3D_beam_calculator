from Section import Section
from Mesh import Mesh
from Line import Line
from Point import Point
from Model import Static
from Load import Displacement, Force, Torque, Pressure
from Material import Material
from Static_results import Static_results as Results


# material definition
E = 200000000000
v = 0.3
material = Material(E=E, v=v)

# I section definition
section = Section()
section.I_shape(0.1, 0.06, 0.02, 0.02)

# geometry definition
point1 = Point(x=0, y=0, z=0, index=1)
point2 = Point(x=0, y=10, z=0, index=2)
point3 = Point(x=-3, y=10, z=0, index=3)
point4 = Point(x=10, y=10, z=0, index=4)


line1 = Line(point1=point1, point2=point2, material=material, section=section, index=1)
line2 = Line(point1=point3, point2=point2, material=material, section=section, index=2)
line3 = Line(point1=point2, point2=point4, material=material, section=section, index=3)

# mesh definition - step 100cm
mesh = Mesh()
mesh.max_element_size(1, [line1, line2, line3])

# boundary conditions definition
force1 = Force(point=point3, direction='y', value=-50000)

displacement1 = Displacement(point=point1, DOF=True)
displacement2 = Displacement(point=point4, uy=0, uz=0, rot_x=0, rot_y=0)

# model definition and solve
model = Static(mesh=mesh, displacement_bc=[displacement1, displacement2], forces_bc=[force1],
               analytical_shear_stresses=True)
model.solve()

# plot results
results = Results(model)

fig1 = results.deformation_3d('uy', show_undeformed=True, show_points=True, show_nodes=True)

fig2 = results.deformation_3d('rotz', show_undeformed=True, show_points=True, show_nodes=True)

fig3 = results.bar_force_2d(option='fy')
fig4 = results.bar_force_2d(option='mz')

fig5 = results.bar_stress_2d(option='ny')
fig6 = results.bar_stress_2d(option='sy')
fig7 = results.bar_stress_3d(option='total')

fig8 = results.section_stress(option='ny', line=line2, length=3)
fig9 = results.section_stress(option='sy', line=line2, length=3)
fig10 = results.section_stress(option='total', line=line2, length=3)

"""
fig1.savefig("Uy_deformation.png")
fig2.savefig("Rot_z_angle.png")
fig3.savefig("Shear_force-local_y_direction.png")
fig4.savefig("Bending_moment-local_z_axis.png")
fig5.savefig("Maximum_Normal_stress_due_to_bending_in_local_y-direction.png")
fig6.savefig("Maximum_Shear_stress_due_to_bending_in_local_y-direction.png")
fig7.savefig("Maximum_von_Misses Stress.png")
fig8.savefig("Section_normal_stress_due_to_bending_in_local_y-direction.png.png")
fig9.savefig("Section_shear_stress_due_to_bending_in_local_y-direction.png")
fig10.savefig("Section_maximum_von_Misses Stress.png")
"""

results.evaluate_all_results()
