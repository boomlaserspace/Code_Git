
from sympy import degree
from generate_xdmf_mesh import *
from os import path
from dolfin import*
from matplotlib import pyplot as plt
from cylindrical_coordinates import * 
from grid_generator_mesh import *
from grid_generator_mesh import _extract_facet_markers

########### INPUT OF CUSTOM MESH #################
xdmf_file, xdmf_facet_marker_file = generate_xdmf_mesh("/Users/davidoexle/Documents/Uni/SoSe22/KontiSim/Code_Git/Grid_Operations/Dose_steht.geo")

mesh = Mesh()
with XDMFFile(xdmf_file) as infile:
    infile.read(mesh)

space_dim = mesh.geometry().dim()
cell_markers = None
try:
    mvc = MeshValueCollection("size_t", mesh, space_dim)
    with XDMFFile(xdmf_file) as infile:
        infile.read(mvc, "cell_markers")
    cell_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)
except Exception:  # pragma: no cover
    pass
    # read facet markers
mvc = MeshValueCollection("size_t", mesh, space_dim - 1)
assert path.exists(xdmf_facet_marker_file)
with XDMFFile(xdmf_facet_marker_file) as infile:
        infile.read(mvc, "facet_markers")

facet_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

dA = Measure("ds", domain=mesh, subdomain_data=facet_markers)
dV = Measure("dx", domain=mesh)
n = FacetNormal(mesh)





##### define constants #######


rayleigh =  10e7 #5056750 #400e5
prandtl = 8.42
nusselt = 124.211
Ra = Constant(rayleigh)
Pr = Constant(prandtl)
Nu = Constant(nusselt)

cfl = 0.1



del_t = 1e-3
delta_t = Expression("dt",degree=2,dt = del_t)
t_end = 10
grav = Constant((0.0,-1.0))
step_t = 0
############### mesh decleration ###################

space_dim = mesh.topology().dim()
n_cells = mesh.num_cells()
hmin = mesh.hmin()

#################### Element Declaration ###################
p_deg = 2
c = mesh.ufl_cell()
v_elem = VectorElement("CG", c, p_deg)
p_elem = FiniteElement("CG", c, p_deg - 1)
T_elem = FiniteElement("CG", c , p_deg)

mixed_elem = MixedElement([v_elem, p_elem, T_elem])
Wh = FunctionSpace(mesh, mixed_elem)
Vh, Ph, Th = Wh.split()


################### BC ############################
v_zero =  Constant((0.0,0.0))


wall_100 = DirichletBC(Vh, v_zero, facet_markers, 100)
wall_101 = DirichletBC(Vh, v_zero, facet_markers, 101)


bc = [wall_100, wall_101] ###### Wall with no slip conditions

################### define Functions ########################
sol = Function(Wh)
vel, press, Temp = split(sol)

sol0 = Function(Wh)
vel0, press0,Temp0 = split(sol0)

sol00 = Function(Wh)
vel00, press00, Temp00 = split(sol00)
test_v ,test_p, test_T = TestFunctions(Wh)

half = Constant(0.5)
two = Constant(2.0)
threehalf = Constant(1.5)

######## Test for the cylindircal Coordinates ######## 
r_coord_inv = Expression("1.0/(x[0]+ tol)", degree=2,tol = 1e-2)
r_coord = Expression("x[0]+tol", degree=2,tol=1e-2)

r_coord_inv = project(r_coord_inv*r_coord,Th.collapse())
File("Koordinate_Radius_Inverse.pvd")<<r_coord_inv

r_coord= project(r_coord,Th.collapse())
File("Koordinate_Radius.pvd")<<r_coord


################# Weak Form ############################
F_p = -(div_cylindrical_vector(vel*r_coord**2)*test_p *dV)

F_v = (
        inner(threehalf*vel-two*vel0 + half*vel00,test_v)*r_coord**2 *dV  # partial time  3*v_n+1 - 4*v_n + 1*v_n-1             LHS
        + delta_t *two*(inner(outer_expend(vel0,test_v), grad_cylindircal_vector(vel0))) *r_coord**2 * dV # first convective term for velocity      LHS
        - delta_t * (inner(outer_expend(vel00,test_v), grad_cylindircal_vector(vel00)))*r_coord**2 * dV # second convective term for velocity     LHS 
        + delta_t * (inner(grad(press),test_v))*r_coord**2 * dV  #Pressure gradient (i) part of sigma                          RHS
        + delta_t * sqrt(Pr/Ra) * (inner(grad(test_v), grad(vel))) *r_coord**2 * dV #(∇ v) * (∇ delv) #impulse diffusity  (ii) part of sigma   RHS
        + delta_t * (Temp) * inner(grav*r_coord**2, test_v) * dV  #Gravityforce   RHS                
    )

F_Temp =( 
            ((threehalf*Temp - two*Temp0 + half*Temp00) * test_T) *r_coord* dV  #partial derivative dT/dt        LHS 
            + delta_t *two*dot(grad(Temp0), vel0) * test_T  *r_coord* dV #convective part 2 v ∇ T         LHS 
            - delta_t *  dot(grad(Temp00), vel00) * test_T *r_coord * dV #convective part - v ∇ T       LHS
            + delta_t * 1./sqrt(Ra*Pr) * inner(grad(Temp), grad(test_T)) *r_coord* dV  # Thermal diffusion (∇ T) * (∇ delT)  RHS  
            - delta_t * Nu/sqrt(Ra*Pr) *(3-Temp)*test_T *r_coord* dA(100)         
    )

F_weak = 2*pi*(F_p + F_v + F_Temp)

J_newton = derivative(F_weak, sol) 
problem = NonlinearVariationalProblem(F_weak, sol, bcs=bc, J=J_newton)
nonlinear_solver = NonlinearVariationalSolver(problem)

################### output folder ###############
output_file = XDMFFile("{0:1.1e}__{1:1.1e}__{2}__{3}results.xdmf".format(int(rayleigh), float(del_t), int(t_end),int(nusselt)))
output_file.parameters["flush_output"] = True
output_file.parameters["functions_share_mesh"] = True
output_file.parameters["rewrite_function_mesh"] = False

vtkfile_velocity =      File('velocity.pvd')
vtkfile_pressure =      File('pressure.pvd')
vtkfile_temperature =   File('temperature.pvd')

step_counter = 0
vmax_array = []
step_array = []
t = 0

while t < t_end:
    step_t += 1/t_end
    step_counter += 1
    t += delta_t.dt 
    nonlinear_solver.solve()

    if step_counter % 10 == 0:
        output_file.write(v_sol, t)
        output_file.write(p_sol, t)
        output_file.write(T_sol, t)

        """
        vtkfile_velocity << (v_sol, t)
        vtkfile_pressure << (p_sol, t)
        vtkfile_temperature << (T_sol , t)
        """
        print("saved file")

    v_sol = sol.split()[0]
    p_sol = sol.split()[1]
    T_sol = sol.split()[2]

    v_sol.rename("v", "Velocity field")
    p_sol.rename("p", "Pressure field")
    T_sol.rename("T", "Temperature field")  

    sol00.assign(sol0)
    sol0.assign(sol)
    

    v_= project(v_sol,Vh.collapse())
    vmax = norm(v_)
    vmax_array.append(vmax)
    step_array.append(step_counter)
    print(f"hmin{hmin}", f"vmax{vmax}",f"CFL{cfl * hmin / vmax}",f"Zeit{t}" )

    print(f"Zeitschritt {delta_t.dt}")

plt.plot(step_array,vmax_array)
plt.show()
