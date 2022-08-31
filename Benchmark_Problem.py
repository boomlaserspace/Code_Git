import json 
from fenics import *
import dolfin as dlfn
from dolfin import inner, grad, div, outer, dot
import numpy as np
import os
import matplotlib.pyplot as plt 
import mpi4py 
os.chdir(os.path.dirname(__file__))

Ra = 3.4e5
Pr = 0.71

cfl = 0.1

##### define constants #######
delta_t = Expression("dt",degree=2,dt = 1e-2)
t_end = 20
grav = Constant((0.0,-1.0))
nx = 32
ny = 140

############### mesh decleration ###################
mesh = RectangleMesh(Point(0.0), Point(1,8),nx, ny)
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
v_lid = Constant((1.0,0.0))

wall_left = DirichletBC(Vh, v_zero, "on_boundary &&  near (x[0] , 0.0 )")
wall_right = DirichletBC(Vh, v_zero, "on_boundary &&  near( x[0] , 1.0 ) ")
wall_up = DirichletBC(Vh, v_zero, "on_boundary && near (x[1] , 8.0 ) ")
wall_down = DirichletBC(Vh, v_zero, "on_boundary &&  near (x[1] , 0.0 ) ")


T_high_left = Constant(0.5)
T_low_right = Constant(-0.5)

low_temperature_bc = DirichletBC(Th, T_low_right, "on_boundary &&  near (x[0], 1.0 )")
high_temperature_bc = DirichletBC(Th, T_high_left,  "on_boundary &&  near (x[0], 0.0)")

bc = [wall_left, wall_right, wall_up , wall_down, low_temperature_bc, high_temperature_bc]

################### define Functions ########################
sol = Function(Wh)
vel, press, Temp = split(sol)

sol0 = Function(Wh)
vel0, press0,Temp0 = split(sol0)

sol00 = Function(Wh)
vel00, press00, Temp00 = split(sol00)

test_v ,test_p, test_T = TestFunctions(Wh)  

p1 = [0.1810, 7.3700]  
p2 = [0.8190, 0.6300]
p3 = [0.1810, 0.6300]
p4 = [0.8190, 7.3700]
p5 = [0.1810, 4.0000]

vx_p1, T_p1= [], []
vx_p1_mean , T_p1_mean = [], []

p_diff_14, p_diff_14_mean = [], []
p_diff_51, p_diff_51_mean = [], []
p_diff_35, p_diff_35_mean = [], []

skewness = []


################# Weak Form ############################
F_p = -(div(vel) * test_p * dx)

F_v = (
        inner(1.5*vel-2*vel0 +0.5*vel00,test_v) * dx  # partial time  3*v_n+1 - 4*v_n + 1*v_n-1             LHS
        + delta_t *2*(inner(outer(vel0,test_v), grad(vel0))) * dx # first convective term for velocity      LHS
        - delta_t * (inner(outer(vel00,test_v), grad(vel00))) * dx # first convective term for velocity     LHS 

        + delta_t * (inner(grad(press),test_v)) * dx  #Pressure gradient (i) part of sigma                          RHS
        + delta_t * np.sqrt(Pr/Ra) * (inner(grad(test_v), grad(vel))) * dx #(∇ v) * (∇ delv) #impulse diffusity  (ii) part of sigma   RHS
        + delta_t * (Temp) * inner(grav, test_v) * dx  #Gravityforce                                                RHS
    )

F_Temp =( 
            ((1.5*Temp - 2*Temp0 + 0.5*Temp00) * test_T) * dx  #partial derivative dT/dt        LHS 
            + delta_t *2*dot(grad(Temp0), vel0) * test_T  * dx #convective part 2 v ∇ T         LHS 
            - delta_t *  dot(grad(Temp00), vel00) * test_T  * dx #convective part - v ∇ T       LHS
            
            + delta_t * 1./np.sqrt(Ra*Pr) * inner(grad(Temp), grad(test_T)) * dx  # Thermal diffusion (∇ T) * (∇ delT)  RHS
    )


F_weak = F_p + F_v + F_Temp

J_newton = derivative(F_weak, sol) 
problem = NonlinearVariationalProblem(F_weak, sol, bcs=bc, J=J_newton)
nonlinear_solver = NonlinearVariationalSolver(problem)

################### output folder ###############
output_file = XDMFFile('Results_Paper.xdmf')
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
    step_counter += 1
    t += delta_t.dt 
    ########### SOLVER ##########
    nonlinear_solver.solve()


    ########### OUTPUT FILES ##########
    if step_counter % 20 == 0:
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
    
    ########### POST PROCESSING STEPS ##########
    v_= project(v_sol,Vh.collapse())
    vmax = norm(v_)
    vmax_array.append(vmax)
    step_array.append(step_counter)

    if vmax > 10e6: 
        plt.xlabel(r'$dt$')
        plt.plot(step_array,skewness ,label =r'$ \epsilon_{12} = T_1 + T_2 $')
        plt.plot(step_array,T_p1_mean,label =r'$ T_{avg.} $')
        plt.plot(step_array,vx_p1_mean,label =r'$ v_{avg.} $')
        plt.plot(step_array,p_diff_14,label =r'$ \Delta p_{14}$')
        plt.legend()

        plt.show()
        raise NameError 



    skewness.append(Temp(np.array(p1)) + Temp(np.array(p2)))

    vx_p1.append(vel[0](np.array(p1)))
    vx_p1_mean.append(np.mean(vx_p1))

    T_p1.append(Temp(np.array(p1)))
    T_p1_mean.append(np.mean(T_p1))

    p_diff_14.append(press(np.array(p1)) - press(np.array(p4)))
    p_diff_14_mean.append(np.mean(p_diff_14))

    p_diff_35.append(press(np.array(p3)) - press(np.array(p5)))
    p_diff_35_mean.append(np.mean(p_diff_35))

    p_diff_51.append(press(np.array(p5)) - press(np.array(p1)))
    p_diff_51_mean.append(np.mean(p_diff_51))




    print(f"hmin{hmin}", f"vmax{vmax}",f"CFL{cfl * hmin / vmax}",f"Zeit{t}" )
    
    if t > 12:
       delta_t.dt = 5e-3

    print(f"Zeitschritt {delta_t.dt}")
   


#print("Schiefsymmetrie",skewness[-1])
#print("Temperatur Punkt1:",T_p1_mean[-1])
#print("v_x1 Punkt1:",vx_p1_mean[-1]) 

#print("pressure difference 14:",p_diff_14_mean[-1])
#print("pressure difference 35:",p_diff_35_mean[-1])
#print("pressure difference 51:",p_diff_51_mean[-1])




dict = { "skewness" : skewness,
          "average_Temperature_p1": T_p1_mean,
          "average_velocity_p1": vx_p1_mean,
          "average_pressure_diff_14": p_diff_14_mean,
          "average_pressure_diff_35": p_diff_35_mean,
          "average_pressure_diff_51": p_diff_51_mean,
          "time_steps": step_array,
}

with open("Auswertung", "w") as fp:   #Pickling
   json.dump(dict, fp) 

