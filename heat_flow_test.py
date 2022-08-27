from sympy import degree
from generate_xdmf_mesh import *
from os import path
from dolfin import*
from matplotlib import pyplot as plt
from cylindrical_coordinates_v2 import* 



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

rayleigh = 3.4e5
prandtl = 0.71
Ra = Constant(rayleigh)
Pr = Constant(prandtl)

cfl = 0.1


##### define constants #######
del_t = 1e-2
delta_t = Expression("dt",degree=2,dt = del_t)
t_end = 1
grav = Constant((0.0,-1.0))
nx = 100
ny = 100

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