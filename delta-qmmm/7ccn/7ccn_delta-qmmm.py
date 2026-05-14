from ash import *
import numpy as np
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
xmlsystemfile = os.path.join(_SCRIPT_DIR, "../../md/7ccn/relaxation/system.xml")
topology_pdb  = os.path.join(_SCRIPT_DIR, "../../md/7ccn/relaxation/system.pdb")
qm_region_pdb = os.path.join(_SCRIPT_DIR, "7ccn_qmregion.pdb")

def get_qm_atoms_from_pdb(pdbfile):
    qm_atoms = []
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                idx = int(parts[1]) - 1  # PDB is 1-indexed, convert to 0-index
                qm_atoms.append(idx)
    return qm_atoms

def esp_at_points(qm_coords, charge_coords, charges):
    """Electrostatic potential at target points from a set of point charges.
    Inputs in Å and elementary charge. Output in atomic units (Hartree/e)."""
    BOHR_PER_A = 1.8897259886
    charges = np.asarray(charges)
    diffs = qm_coords[:, None, :] - np.asarray(charge_coords)[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return (charges[None, :] / (dists * BOHR_PER_A)).sum(axis=1)

def save_results_to_npz(frag, dftmm, xtbmm, esp, filename):
    np.savez(
        filename,
        qm_coords = frag.coords[dftmm.qmatoms],
        qm_elems = np.array(frag.elems)[dftmm.qmatoms],
        delta_e = dftmm.QMenergy - xtbmm.QMenergy,
        delta_f = -(dftmm.QM_PC_gradient[dftmm.qmatoms] 
                  - xtbmm.QM_PC_gradient[dftmm.qmatoms]),
        dft_e = dftmm.QMenergy,
        dft_f = -dftmm.QM_PC_gradient[dftmm.qmatoms],
        esp = esp
    )        

def cleanup_results_files():
    for file in os.listdir("."):
        if file.startswith("xtb") or file.startswith("gcp") or file.startswith("pyscf"):
            os.remove(file)
        if file in ["pcgrad", "energy", "charges"]:
            os.remove(file)

if __name__ == "__main__":
    pdbfile = sys.argv[1]
    snapshot_number = pdbfile.split("/")[-1].split(".")[0]
    frag = ash.Fragment(pdbfile = pdbfile)
    qm_atoms = get_qm_atoms_from_pdb(qm_region_pdb)

    # MM theory built from the System XML (carries the 12-6-4 forces) + topology PDB
    mm = OpenMMTheory(
        xmlsystemfile = xmlsystemfile, 
        pdbfile = topology_pdb,
        periodic = True, 
        autoconstraints = None, 
        rigidwater = False
    )

    qm_xtb = xTBTheory(xtbmethod="GFN2")

    qm_pyscf = PySCFTheory(
        scf_type = "RKS", 
        functional = "r2scan",
        basis = "def2-mtzvpp", 
        auxbasis = "def2-universal-jfit",
        ecp = {"Lu" : "def2-ecp"},
        densityfit = True, 
        platform = "GPU",
        write_chkfile_name = None,
        noautostart = True, 
        guess = "atom",
        level_shift = 0,
        damping = 0.5,
        scf_maxiter = 150,
        radii = None
    )
    gcp_corr = gcpTheory(functional = "r2SCAN-3c")
    d4_corr = DFTD4Theory(functional = "r2SCAN-3c")
    qm_r2scan3c = WrapTheory(theories = [qm_pyscf, gcp_corr, d4_corr])

    xtbmm = QMMMTheory(
        fragment = frag, 
        qm_theory = qm_xtb,
        mm_theory = mm, 
        qmatoms = qm_atoms,
        embedding = "elstat", 
        qm_charge = -1,
        qm_mult = 1, 
        printlevel = 2
    )

    dftmm = QMMMTheory(
        fragment = frag, 
        qm_theory = qm_r2scan3c,
        mm_theory = mm, 
        qmatoms = qm_atoms,
        embedding = "elstat", 
        qm_charge = -1,
        qm_mult = 1, 
        printlevel = 2
    )

    # --- run with gradient ---
    xtbmm.run(
        current_coords = frag.coords, 
        elems = frag.elems,
        Grad = True
    )

    dftmm.run(
        current_coords = frag.coords, 
        elems = frag.elems,
        Grad = True
    )

    esp_at_qm = esp_at_points(
        frag.coords[dftmm.qmatoms], 
        dftmm.pointchargecoords, 
        dftmm.pointcharges
    )

    save_results_to_npz(
        frag, dftmm, xtbmm, esp_at_qm,
        filename = f"{snapshot_number}.npz"
    )

    cleanup_results_files()
