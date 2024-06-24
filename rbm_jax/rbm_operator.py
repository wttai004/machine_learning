from model import Model
from rbm import RBM
import numpy as np
import random

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

    def add_Sx(self, i, j, J = 1):
        operator = self.generate_identity()
        operator[i, j, 0, 0] = 0
        operator[i, j, 1, 1] = 0
        operator[i, j, 0, 1] = 1/2 * J 
        operator[i, j, 1, 0] =  1/2 * J
        self.onesiteoperators.append(operator)
    
    def add_Sy(self, i, j, J = 1):
        operator = self.generate_identity()
        operator[i, j, 0, 0] = 0
        operator[i, j, 1, 1] = 0
        operator[i, j, 0, 1] = -1j * 1/2 * J
        operator[i, j, 1, 0] =  1j * 1/2 * J
        self.onesiteoperators.append(operator)

    def add_Sz(self, i, j, J = 1):
        operator = self.generate_identity()
        operator[i, j, 0, 0] = 1/2 * J
        operator[i, j, 1, 1] = -1/2 * J
        self.onesiteoperators.append(operator)

    def add_SzSz(self, i1, j1, i2, j2, J = 1):
        result = []
        operator = self.generate_identity()
        operator[i1, j1, 0, 0] = 1/2 * J
        operator[i1, j1, 1, 1] = -1/2 * J
        result.append(operator)
        operator = self.generate_identity()
        operator[i2, j2, 0, 0] = 1/2 * J
        operator[i2, j2, 1, 1] = -1/2 * J
        result.append(operator)
        self.twositeoperators.append(result)

    def add_SpSm(self, i1, j1, i2, j2, J = 1):
        result = []
        operator = self.generate_identity()
        operator[i1, j1, 0, 0] = 0
        operator[i1, j1, 1, 1] = 0
        operator[i1, j1, 0, 1] = 1/2 * J
        result.append(operator)
        operator = self.generate_identity()
        operator[i2, j2, 0, 0] = 0
        operator[i2, j2, 1, 1] = 0
        operator[i2, j2, 1, 0] = J
        result.append(operator)
        self.twositeoperators.append(result)

    def add_SmSp(self, i1, j1, i2, j2, J = 1):
        result = []
        operator = self.generate_identity()
        operator[i1, j1, 0, 0] = 0
        operator[i1, j1, 1, 1] = 0
        operator[i1, j1, 1, 0] = 1/2 * J
        result.append(operator)
        operator = self.generate_identity()
        operator[i2, j2, 0, 0] = 0
        operator[i2, j2, 1, 1] = 0
        operator[i2, j2, 0, 1] = J
        result.append(operator)
        self.twositeoperators.append(result)

    def add_SdotS_interaction(self, i1, j1, i2, j2, J = 1):
        """ 
        Add the S(r1).S(r2) interaction between spins at r1 and r2, with strength J

        The Hamiltonian represents the entries 1/2(S+(r1).S-(r2) + S-(r1).S+(r2)) + Sz(r1).Sz(r2)
        """
        self.add_SzSz(i1, j1, i2, j2, J = J)
        self.add_SpSm(i1, j1, i2, j2, J = J)
        self.add_SmSp(i1, j1, i2, j2, J = J)
    
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
        for operator_combo in self.twositeoperators:
            temp = np.copy(spin2)
            for operator in operator_combo:
                temp = np.einsum("ijkl, ijk->ijl", operator, temp)
            #print(spin1, temp)
            twositeresult += self.model.vdot(spin1, temp)
        return onesiteresult + twositeresult
   
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

if __name__ == "__main__":
    test_J1 = False
    if test_J1:
        # Test the J1 Hamiltonian on 2x1 lattice
        model = Model(2,1)
        J1_Op = set_J1_Hamiltonian(model)
        #print(f"J1 Hamiltonian: {J1_Op.onesiteoperators}, {J1_Op.twositeoperators}")
        spin1 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
        spin2 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
        print(f"Hamiltonian expectation: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
        spin2[1, 0] = [0,1]
        print(f"Flipping a spin in s Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
        spinud = spin1.copy()
        spinud[1, 0] = [0, 1]
        spindu = spin1.copy()
        spindu[0, 0] = [0, 1]
        singlet_energy = (J1_Op.vdot(spinud, spinud)-J1_Op.vdot(spindu, spinud)-J1_Op.vdot(spinud, spindu)+J1_Op.vdot(spindu, spindu))/2
        triplet_energy = (J1_Op.vdot(spinud, spinud)+J1_Op.vdot(spindu, spinud)+J1_Op.vdot(spinud, spindu)+J1_Op.vdot(spindu, spindu))/2
        print(f"J1 Hamiltonian on singlet state gives: <spin1|H|spin2>={singlet_energy}")
        print(f"J1 Hamiltonian on triplet state gives: <spin1|H|spin2>={triplet_energy}")

        # Test the J1 Hamiltonian on 4x4 lattice
        model = Model(4, 4)
        spin1 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
        spin2 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
        J1_Op = set_J1_Hamiltonian(model)
        print(f"J1 Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
        spin2[0,0] = [0, 1]
        print(f"Flipping a corner spin in s Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
        print(f"Energy is now: <spin2|H|spin2>={J1_Op.vdot(spin2, spin2)}")
        spin2[2,2] = [0, 1]
        print(f"Flipping a middle spin in s Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
        print(f"Energy is now: <spin2|H|spin2>={J1_Op.vdot(spin2, spin2)}")

    test_h = False
    if test_h:
        model = Model(4,4)
        Ham = set_h_Hamiltonian(model, h = 1)
        #Ham += set_J1_Hamiltonian(model, J = 1)
        spin1 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
        print(f"Hamiltonian expectation: <spin1|H|spin2>={Ham.vdot(spin1, spin1)}")
        spin1[1, 0] = [0,1]
        print(f"Flipping a spin in s Hamiltonian gives: <spin1|H|spin2>={Ham.vdot(spin1, spin1)}")
        spin1[3, 0] = [0,1]
        print(f"Flipping a spin in s Hamiltonian gives: <spin1|H|spin2>={Ham.vdot(spin1, spin1)}")

    # Test the SzSz expectation value
    test_SzSz = False
    if test_SzSz == True:
        model = Model(2,2)
        rbm = RBM(model)
        average_expectations = [rbm.expectation_value(SzSz_(0,0,0,1,model), np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])) for _ in range(40)]
        print(f"The average expectation <SzSz|ud__|SzSz> is {np.mean(average_expectations)} with standard deviation {np.std(average_expectations)}")
        average_expectations = [rbm.expectation_value(SzSz_(0,0,0,1,model), np.array([[[0, 1], [0, 1]], [[0, 1], [1, 0]]])) for _ in range(40)]
        print(f"The average expectation  <SzSz|dd__|SzSz> is {np.mean(average_expectations)} with standard deviation {np.std(average_expectations)}")
        rbm = RBM(model)
        batch = rbm.create_batch(200) #This uses evaluate_dummy, which gives 2/3 for spin up and 1/3 for spin down 
        average_expectations = [rbm.expectation_value(SzSz_(0,0,0,1,model), batch[i]) for i in range(len(batch))]
        print(f"The average expectation for a 2:1 mixed state is {np.mean(average_expectations)})")

    # Test the model expectation
    test_expectation = True
    if test_expectation:
        model = Model(2,3)
        rbm = RBM(model)

        N = 100

        #create a batch
        batch = rbm.create_batch(N)
        # Implement a Hamiltonian

        Ham =  set_J1_Hamiltonian(model, J = 1)#set_h_Hamiltonian(model, h = 4)

        Szs = set_h_Hamiltonian(model, h = 1)

        print(f"the spin expectation value is {rbm.expectation_value_batch(Szs, batch)}")

        def calculate_Sz_expectation_brute_force(spins):
            return  np.mean(np.sum(batch[:, :, :, 0]/2 - batch[:, :, :, 1]/2, axis=(1, 2)))

        print(f"the naive spin expectation value is {calculate_Sz_expectation_brute_force(batch)}")
        