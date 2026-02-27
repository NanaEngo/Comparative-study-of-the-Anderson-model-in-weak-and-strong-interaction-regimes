import numpy as np

try:
    import quimb.tensor as qtn

    QUIMB_AVAILABLE = True
except ImportError:
    QUIMB_AVAILABLE = False


def _construct_mpo_from_bcf(time_grid, correlation_func, bond_dim=10):
    if not QUIMB_AVAILABLE:
        return None

    n_steps = len(time_grid)
    dt = time_grid[1] - time_grid[0]

    t_row = time_grid[:, None]
    t_col = time_grid[None, :]
    delta_t = np.abs(t_row - t_col)

    C = np.array(correlation_func(delta_t))
    C = np.nan_to_num(C, nan=0.0)

    U, S, Vh = np.linalg.svd(C, full_matrices=False)

    eff_bond_dim = min(bond_dim, n_steps)
    U_trunc = U[:, :eff_bond_dim]
    S_trunc = S[:eff_bond_dim]
    Vh_trunc = Vh[:eff_bond_dim, :]

    mpo = qtn.MPO_identity(n_steps, phys_dim=2)

    for i, t in enumerate(mpo.tensors):
        mode_amplitude = np.sum(U_trunc[i, :] * np.sqrt(S_trunc))
        mode_coupling = np.sum(Vh_trunc[:, i] * np.sqrt(S_trunc))

        scale_factor = np.abs(mode_amplitude * np.exp(-dt * np.abs(mode_coupling)))
        if scale_factor < 1e-12:
            scale_factor = 1e-12

        t.modify(data=t.data * scale_factor)

    return mpo


if __name__ == "__main__":
    if QUIMB_AVAILABLE:
        tg = np.linspace(0, 10, 100)

        def corr(t):
            return np.exp(-t)

        mpo = _construct_mpo_from_bcf(tg, corr)
        print("MPO created:", mpo)
    else:
        print("Quimb not available")
