   
def Sx_(i, j, model, J = 1):
    """
    Create the Sx operator at site i,j
    """
    result = Operator(model)
    result.add_Sx(i, j)
    return result

def Sy_(i, j, model):
    """
    Create the Sy operator at site i,j
    """
    result = Operator(model)
    result.add_Sy(i, j)
    return result

def Sz_(i, j, model):
    """
    Create the Sz operator at site i,j
    """
    result = Operator(model)
    result.add_Sz(i, j)
    return result

def SzSz_(i1, j1, i2, j2, model):
    """
    Create the SzSz operator at sites i1,j1 and i2,j2
    """
    result = Operator(model)
    result.add_SzSz(i1, j1, i2, j2)
    return result

def set_h_Hamiltonian(model, h = 1):
    """
    Sets the Hamiltonian to be the h field
    """
    result = Operator(model)
    for i in range(model.L1):
        for j in range(model.L2):
            result.add_Sz(i, j, h)
    return result

def set_J1_Hamiltonian(model, J = 1):
    """
    Sets the Hamiltonian to be the J1 Heisenberg model
    """
    result = Operator(model)
    for i in range(model.L1-1):
        for j in range(model.L2):
            result.add_SdotS_interaction(i, j, i+1, j, J)  
    for i in range(model.L1):
        for j in range(model.L2-1):      
            result.add_SdotS_interaction(i, j, i, j+1, J)  
    return result

def set_J2_Hamiltonian(model, J = 1):
    """
    Sets the Hamiltonian to be the J1 Heisenberg model
    """
    result = Operator(model)
    for i in range(model.L1-1):
        for j in range(model.L2):
            result.add_SdotS_interaction(i, j, i+1, j+1, J)  
    for i in range(model.L1):
        for j in range(model.L2-1):        
            result.add_SdotS_interaction(i, j, i+1, j+1, J)  
    return result