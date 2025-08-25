import numpy as np

def topological_neighbors(L, agent):
    """ Returns the neighbors of a particular agent using the graph Laplacian

    L: NxN numpy array (representing the graph Laplacian)
    agent: int (agent: 0 - N-1)

    -> 1xM numpy array (with M neighbors)
    """
    #Check user input types
    assert isinstance(L, np.ndarray), "In the topological_neighbors function, the graph Laplacian (L) must be a numpy ndarray. Recieved type %r." % type(L).__name__
    assert isinstance(agent, int), "In the topological_neighbors function, the agent number (agent) must be an integer. Recieved type %r." % type(agent).__name__
    
    #Check user input ranges/sizes
    assert agent >= 0, "In the topological_neighbors function, the agent number (agent) must be greater than or equal to zero. Recieved %r." % agent
    assert agent <= L.shape[0], "In the topological_neighbors function, the agent number (agent) must be within the dimension of the provided Laplacian (L). Recieved agent number %r and Laplactian size %r by %r." % (agent, L.shape[0], L.shape[1])

    row = L[agent, :]
    row[agent]=0
    # Since L = D - A
    return np.where(row != 0)[0]