import numpy as np

def compute_corr_sum(corrs, delta_x, delta_y, Lx, Ly, pbc = False):
    """
    Compute the correlation function between two operators separated by a distance delta_x, delta_y

    Parameters:
    corrs: list of correlation functions psi.correlation_function("Nu", "Nu")    
           corrs[i][j] is the correlation function between the i-th and j-th sites
    delta_x: distance in x direction
    delta_y: distance in y direction
    Lx: number of sites in x direction
    Ly: number of sites in y direction

    Return:
    result: sum of the correlation function over the lattice
    """
    def index(i,j):
        return i * Ly + j

    # compute <n_is n_js>
    result = 0
    for i in range(Lx):
        for j in range(Ly):
            if pbc == False:
                if i + delta_x >= Lx or j + delta_y >= Ly:
                    continue
            result += corrs[index(i,j)][index((i+delta_x) % Lx,(j+delta_y) % Ly)]
    return result

def compute_corr_results(corr_type, psi, Lx, Ly, pbc = False):
    """
    Compute the correlation results for a specific type of correlation function.

    Parameters:
    corr_type: tuple of (op1, op2) wher
    op1, op2: string representing the operator type
    psi: object containing the correlation function method
    Lx: number of sites in x direction
    Ly: number of sites in y direction

    Returns:
    corr_result: 2D numpy array of correlation sums
    """
    corrs = psi.correlation_function(*corr_type)
    corr_result = np.array([[compute_corr_sum(corrs, i, j, Lx, Ly, pbc = pbc) for j in range(Ly)] for i in range(Lx)])
    return corr_result


if __name__ == "__main__":
    import tenpy
    from tenpy.models.lattice import Square
    from tenpy.models import CouplingMPOModel
    from tenpy.networks.site import SpinHalfFermionSite, spin_half_species
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
    from tenpy.algorithms.mps_common import DensityMatrixMixer
    from timeit import default_timer as timer

    class FermiHubbardSquare(CouplingMPOModel):
        
        def init_sites(self, model_params):
            cons_N = model_params.get('cons_N', 'N', str)
            cons_Sz = model_params.get('cons_Sz', 'Sz', str)
            site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
            return site

        def init_lattice(self, model_params):
            site = self.init_sites(model_params)
            Lx = model_params.get('Lx', 4)
            Ly = model_params.get('Ly', 4)
            bc_x = model_params.get('bc_x', 'open', str)
            bc_y = model_params.get('bc_y', 'open', str)
            return Square(Lx=Lx, Ly=Ly, site=site, bc = [bc_x, bc_y])

        def init_terms(self, model_params):
            # read out parameters
            t = model_params.get("t", 1.) 
            Uv = model_params.get("U", 1.)
            m = model_params.get("m", 1)
            # add terms
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(Uv, u, 'NuNd')
                self.add_onsite(m, u, "Nu")
                self.add_onsite(-m, u, "Nd")
            for i1, i2, dx in self.lat.pairs["nearest_neighbors"]:
                print(i1, i2, dx)
                self.add_coupling(t, i1, "Cdu", i2, "Cu", dx, plus_hc=True)
                self.add_coupling(-t, i1, "Cdd", i2, "Cd", dx, plus_hc=True)
                if np.array_equal(dx, [1,0]):
                    self.add_coupling(1j * t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                    self.add_coupling(1j * t, i1, "Cdd", i2, "Cu", dx, plus_hc=True)
                if np.array_equal(dx, [0,1]):
                    self.add_coupling(t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                    self.add_coupling(-t, i1, "Cdd", i2, "Cu", dx, plus_hc=True)        

    L = 2
    Lx = 2
    Ly = 2
    t = 1.0
    U = 1.0
    m = 5.0
    pbc = False

    # Create the Fermi-Hubbard Model
    model_params = {
        't': t,                 # Nearest-neighbor hopping strength
        'U': U,                 # On-site Hubbard interaction
        'm': m,                 # Mass gap
        'Lx': Lx,
        'Ly': Ly,
        'cons_Sz': None,
        'cons_N': 'N',
        'bc_x': 'periodic' if pbc else 'open',
        'bc_y': 'periodic' if pbc else 'open'
    }

    dmrg_params = {
        "mixer": DensityMatrixMixer,
        "mixer_params": {
            "amplitude": 0.3,
            "decay": 2,
            "disable_after": 50
        },
        "trunc_params": {
            "chi_max": 500, #bond dimension
            "svd_min": 1*10**-10
        },
        "max_E_err": 0.0001, #energy convergence step threshold
        "max_S_err": 0.0001, #entropy convergence step threshold
        "max_sweeps": 2000  #may or may not be enough to converge
    }


    model = FermiHubbardSquare(model_params)
    H_mpo = model.calc_H_MPO()
    print(f'MPO bond dimensions: {H_mpo.chi}')

    product_state = ['empty' if n < Lx * Ly // 2 else 'up' for n in range(Lx * Ly)]
    #product_state = ["up" for n in range(Lx * Ly)]

    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS, dtype=np.complex128)

    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)

    E0, psi = engine.run()
    print(f'Ground state energy: {E0}')

    correlation_types = {
        "uu": ("Nu", "Nu"),
        "ud": ("Nu", "Nd"),
        "dd": ("Nd", "Nd")
    }

    # Compute and store correlation results
    corrs_results = {}
    for label, corr_type in correlation_types.items():
        corrs_results[label] = compute_corr_results(corr_type, psi, Lx, Ly)

    print(corrs_results)