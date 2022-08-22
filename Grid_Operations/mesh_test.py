
from generate_xdmf_mesh import *
from os import path
import dolfin as dlfn

xdmf_file, xdmf_facet_marker_file = generate_xdmf_mesh("/home/david/Documents/SoSe22/KontiSim/Code/Grid_Operations/Dose_steht.geo")



mesh = dlfn.Mesh()
with dlfn.XDMFFile(xdmf_file) as infile:
    infile.read(mesh)

space_dim = mesh.geometry().dim()
cell_markers = None
try:
    mvc = dlfn.MeshValueCollection("size_t", mesh, space_dim)
    with dlfn.XDMFFile(xdmf_file) as infile:
        infile.read(mvc, "cell_markers")
    cell_markers = dlfn.cpp.mesh.MeshFunctionSizet(mesh, mvc)
except Exception:  # pragma: no cover
    pass
    # read facet markers
mvc = dlfn.MeshValueCollection("size_t", mesh, space_dim - 1)
assert path.exists(xdmf_facet_marker_file)
with dlfn.XDMFFile(xdmf_facet_marker_file) as infile:
        infile.read(mvc, "facet_markers")

facet_markers = dlfn.cpp.mesh.MeshFunctionSizet(mesh, mvc)

# class for periodic boundary conditions
class PeriodicBoundary(dlfn.SubDomain):
        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < dlfn.DOLFIN_EPS and x[0] > - dlfn.DOLFIN_EPS and on_boundary)

        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - 1
            y[1] = x[1]

    # instance for periodic boundary conditions
pbc = PeriodicBoundary()

    # == Element Formulation ======================================================
c = mesh.ufl_cell()
v_elem = dlfn.VectorElement("CG", c, 2)
p_elem = dlfn.FiniteElement("CG", c, 2 - 1)
mixed_elem = dlfn.MixedElement([v_elem, p_elem])

Wh = dlfn.FunctionSpace(mesh, mixed_elem)

n_dofs = Wh.dim()


    # == Boundary Conditions ======================================================
Vh, Ph = Wh.split()
null_vector = dlfn.Constant((0., ) * space_dim)
v0 = dlfn.Expression(("1.0", "0.0"), degree=1)
v1 = dlfn.Expression(("1.5", "0.0"), degree=1)

dirichlet_bcs = []
dirichlet_bcs.append(dlfn.DirichletBC(Vh, null_vector, facet_markers, 100))
dirichlet_bcs.append(dlfn.DirichletBC(Vh, v1, facet_markers, 101))

    # == Test and Trial Functions =================================================
(del_v, del_p) = dlfn.TestFunctions(Wh)
sol = dlfn.Function(Wh)
sol_v, sol_p = dlfn.split(sol)


    # == Surface and Volume Element ===============================================
dA = dlfn.Measure("ds", domain=mesh, subdomain_data=facet_markers)
dV = dlfn.Measure("dx", domain=mesh)
n = dlfn.FacetNormal(mesh)


    # == Nonlinear Form ===========================================================
F_rho = - del_p * dlfn.div(sol_v) * dV
F_v = (dlfn.inner(dlfn.outer(del_v, sol_v), dlfn.grad(sol_v)) - sol_p * dlfn.div(del_v) + dlfn.inner(dlfn.grad(sol_v), dlfn.grad(del_v))) * dV 

F = F_v + F_rho

    # == Newton's Method ==========================================================
J_newton = dlfn.derivative(F, sol)
problem = dlfn.NonlinearVariationalProblem(F, sol, bcs=dirichlet_bcs, J=J_newton)
nonlinear_solver = dlfn.NonlinearVariationalSolver(problem)
nonlinear_solver.solve()
    
    # == Save Data as PVD =========================================================
v_out = sol.split()[0]
p_out = sol.split()[1]
    
    
v_out.rename("v", "Spatial velocity field")
p_out.rename("p", "Spatial pressure function")

dlfn.File('velocity.pvd') << v_out
dlfn.File('pressure.pvd') << p_out














