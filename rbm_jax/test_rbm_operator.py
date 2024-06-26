

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
        