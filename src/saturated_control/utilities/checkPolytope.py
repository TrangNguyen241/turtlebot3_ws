import numpy as np
# Using U constraint as a ball (radius = ru)
# def check_u_inputs(ru, u):
#     ''' Check if the control input is within the input contraint
#     Parameters:
#     u_contraints: input contraint is represented by polytope
#     u: control signal 
#     Return:
#     True if contraints are satisfied, False otherwise. 
#     '''
#     return (np.linalg.norm(u) <= ru)

# Using constraint as a Polytope
def check_u_inputs(U_input, u):
    ''' Check if the control input is within the input contraint
    Parameters:
    u_contraints: input contraint is represented by polytope
    u: control signal 
    Return:
    True if contraints are satisfied, False otherwise. 
    '''
    return np.all((U_input.A @ u) <= U_input.b)

