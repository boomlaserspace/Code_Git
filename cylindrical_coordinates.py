from generate_xdmf_mesh import *
from os import path
from dolfin import*
from dolfin import grad
from matplotlib import pyplot as plt
from ufl import as_tensor
import numpy as np

################# ABLEITUNGSOPERATOREN FÜR SKALARE ###################

def grad_cylindrical_scalar(scal): #### Hier wird ein Vektor zurückgegeben


    return None



################# ABLEITUNGSOPERATOREN FÜR VEKTOREN ###################

def div_cylindrical_vector(vec):  ##### hier wird ein skalar zurückgegeben 
    
    r_coord_inv = Expression("1.0/(x[0]+ tol)", degree=2,tol = 1e-2)

    scalar_cylindrical_grad = as_tensor( vec[0].dx(0) + r_coord_inv*vec[0] + vec[1].dx(1))

    return scalar_cylindrical_grad



def grad_cylindircal_vector(vec): ##### hier wird ein Tensor zurückgegeben 
    r_coord_inv = Expression("1.0/(x[0]+ tol)", degree=2,tol = 1e-2)

    tensor_clyindrical_grad = as_tensor( 
        [[ vec[0].dx(0), 0, vec[0].dx(1)], 
         [0,r_coord_inv * vec[1] ,0],
         [ vec[1].dx(0), 0, vec[1].dx(1)]] 
        )
    return tensor_clyindrical_grad

def outer_expend(vec1,vec2):  ### 2D vektoren werden erweiter sodass es passt am Ende

    expanded_tensor = as_tensor( 
        [[ vec1[0]*vec2[0], 0, vec1[0]*vec2[1]], 
         [ 0,0,0],
         [ vec1[1]*vec2[0], 0, vec1[1]*vec2[1]]] 
        )
    return expanded_tensor















   
