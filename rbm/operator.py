import model
import numpy as np

class Operator:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.L1 = model.L1
        self.L2 = model.L2
        self.onesiteoperators = []
        self.twositeoperators = []

    def generate_identity(self):
        result = np.zeros((self.L1, self.L2, 2, 2), dtype = complex)
        #adding the identity operators
        for i in range(self.L1):
            for j in range(self.L2):
                result[i, j, 0, 0] = 1  
                result[i, j, 1, 1] = 1
        return result

    def add_Sx(self, i, j):
        operator = self.generate_identity()
        operator[i, j, 0, 0] = 0
        operator[i, j, 1, 1] = 0
        operator[i, j, 0, 1] = 1/2
        operator[i, j, 1, 0] =  1/2
        self.onesiteoperators.append(operator)
    
    def add_Sy(self, i, j):
        operator = self.generate_identity()
        operator[i, j, 0, 0] = 0
        operator[i, j, 1, 1] = 0
        operator[i, j, 0, 1] = -1j * 1/2
        operator[i, j, 1, 0] =  1j * 1/2
        self.onesiteoperators.append(operator)

    def add_Sz(self, i, j):
        operator = self.generate_identity()
        operator[i, j, 0, 0] = 1/2
        operator[i, j, 1, 1] = -1/2
        self.onesiteoperators.append(operator)

    def add_SzSz(self, i1, j1, i2, j2):
        result = []
        operator = self.generate_identity()
        operator[i1, j1, 0, 0] = 1/2
        operator[i1, j1, 1, 1] = -1/2
        result.append(operator)
        operator = self.generate_identity()
        operator[i2, j2, 0, 0] = 1/2
        operator[i2, j2, 1, 1] = -1/2
        result.append(operator)
        self.twositeoperators.append(result)

    def add_SdotS_interaction(self, i1, j1, i2, j2):
        """ 
        Add the S(r1).S(r2) interaction between spins at r1 and r2, with strength J

        The Hamiltonian represents the entries 1/2(S+(r1).S-(r2) + S-(r1).S+(r2)) + Sz(r1).Sz(r2)
        """
        self.twositeoperators[i1, j1, i2, j2, 0, 0, 0, 0] += 1/4 
        self.twositeoperators[i1, j1, i2, j2, 0, 1, 0, 1] += -1/4 
        self.twositeoperators[i1, j1, i2, j2, 0, 1, 1, 0] += 1/2
        self.twositeoperators[i1, j1, i2, j2, 1, 0, 1, 0] += -1/4 
        self.twositeoperators[i1, j1, i2, j2, 1, 0, 0, 1] += 1/2
        self.twositeoperators[i1, j1, i2, j2, 1, 1, 1, 1] += 1/4 
    
    def __add__(self, other):
        #Add two operators
        assert self.L1 == other.L1 and self.L2 == other.L2, "Cannot add operators of different sizes"
        newObject = Operator(self.model)
        newObject.onesiteoperators = self.onesiteoperators + other.onesiteoperators
        newObject.twositeoperators = self.twositeoperators + other.twositeoperators
        return newObject
  

    def vdot(self, spin1, spin2):
        #return the vector product between the operator and the spin
        assert spin1.dtype == int and spin2.dtype == int, f"Hdot cannot handle spins of type {spin1.dtype}. Please convert this to int"
        assert np.all(np.sum(spin1, axis = -1) == 1), "spin1 is not properly normalized"
        assert np.all(np.sum(spin2, axis = -1) == 1), f"spin2 is not properly normalized for {spin2}"
        
        onesiteresult = 0
        for operator in self.onesiteoperators:
            onesiteresult += self.model.vdot(spin1, np.einsum("ijkl, ijk->ijl", operator, spin2))
        twositeresult = 0
        temp = np.copy(spin2)
        for operator_combo in self.twositeoperators:
            for operator in operator_combo:
                temp = np.einsum("ijkl, ijk->ijl", operator, temp)
            twositeresult += self.model.vdot(spin1, temp)
        return onesiteresult + twositeresult
   
def Sx_(i, j, model):
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

if __name__ == "__main__":
    print("Hello world")