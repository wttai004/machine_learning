import netket as nk
import netket.experimental as nkx
import scipy


def get_ed_data(H, k = 2):
    # Convert the Hamiltonian to a sparse matrix
    sp_h = H.to_sparse()

    eig_vals, eig_vecs = scipy.sparse.linalg.eigsh(sp_h, k=k, which="SA")

    sorted_indices = eig_vals.argsort()
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]

    E_gs = eig_vals[0]

    print("Exact ground state energy:", E_gs)
    return eig_vals, eig_vecs