from ash import *
from math import floor

pdbfile = "../../md/7ccn/production/snapshots/snapshot_0000.pdb"
xmlsystemfile = "../../md/7ccn/relaxation/system.xml"
topology_pdb = "../../md/7ccn/relaxation/system.pdb"
qm_region_pdb = "7ccn_qmregion.pdb"
frag = ash.Fragment(pdbfile=pdbfile)

def get_qm_atoms_from_pdb(pdbfile):
    qm_atoms = []
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                idx = int(parts[1]) - 1  # PDB is 1-indexed, convert to 0-index
                qm_atoms.append(idx)
    return qm_atoms

def get_lattice_vectors_from_pdb(pdbfile):
    """Return 3x3 lattice matrix (Angstrom) from CRYST1 record. Each row is a vector."""
    import numpy as np
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith("CRYST1"):
                parts = line.split()
                a, b, c = float(parts[1]), float(parts[2]), float(parts[3])
                alpha = np.deg2rad(float(parts[4]))
                beta  = np.deg2rad(float(parts[5]))
                gamma = np.deg2rad(float(parts[6]))
                ax = a
                bx = b * np.cos(gamma)
                by = b * np.sin(gamma)
                cx = c * np.cos(beta)
                cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
                cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
                return np.array([[ax, 0.0, 0.0],
                                 [bx, by,  0.0],
                                 [cx, cy,  cz]])
    raise ValueError("No CRYST1 record found in PDB file.")

def get_rcut_hcore_ewald(frag, qm_atoms, lattice_vectors):
    qm_coords = np.array([frag.coords[i] for i in qm_atoms])
    qm_center = qm_coords.mean(axis=0)
    r = qm_coords - qm_center  # relative positions

    # Lower bound: farthest QM atom from centroid
    lower = np.max(np.linalg.norm(r, axis=1))

    # Upper bound: nearest periodic image of any QM atom
    upper = np.inf
    for atom_r in r:
        for vec in lattice_vectors: 
            for sign in (+1, -1):
                upper = min(upper, np.linalg.norm(atom_r + sign * vec))

    assert lower < upper, f"QM region too large for box: lower={lower:.2f}, upper={upper:.2f} Å"

    # Pick midpoint, rounded conservatively
    rcut_hcore = int((lower + upper) / 2)
    rcut_ewald = int(floor(np.min(np.diag(lattice_vectors)) / 2))

    return rcut_hcore, rcut_ewald

qm_atoms = get_qm_atoms_from_pdb(qm_region_pdb)
print(f"QM atoms: {qm_atoms}")

lattice_vectors = get_lattice_vectors_from_pdb(pdbfile)
print(f"Lattice vectors:\n{lattice_vectors}")

rcut_hcore, rcut_ewald = get_rcut_hcore_ewald(frag, qm_atoms, lattice_vectors)
print(f"Chosen rcut_hcore = {rcut_hcore} Å, rcut_ewald = {rcut_ewald} Å")

# MM theory built from the System XML (carries the 12-6-4 forces) + topology PDB
mm = ash.OpenMMTheory(
    xmlsystemfile=xmlsystemfile, 
    pdbfile=topology_pdb,
    periodic=True, 
    autoconstraints=None, 
    rigidwater=False
)

# 0.2 Å makes zeta = 7 Bohr^{-2}, effectively a point charge
radii = np.full(mm.system.getNumParticles(), 0.2) 

qm_xtb = ash.xTBTheory(xtbmethod="GFN2")

qm_pyscf = ash.PySCFTheory(
    scf_type="RKS", 
    functional="r2scan",
    basis="def2-mtzvpp", 
    auxbasis="def2-universal-jfit",
    ecp={"Lu": "def2-ecp"},
    densityfit=True, 
    platform="GPU",
    #PBC_lattice_vectors=lattice_vectors,
    rcut_hcore=rcut_hcore, 
    rcut_ewald=rcut_ewald,
    write_chkfile_name=None,
    noautostart=True, 
    guess="atom",
    level_shift=0.2,
    damping=0.5,
    scf_maxiter=150,
    radii=None
)
gcp_corr = gcpTheory(functional="r2SCAN-3c")
d4_corr = DFTD4Theory(functional="r2SCAN-3c")
qm_r2scan3c = WrapTheory(theories=[qm_pyscf, gcp_corr, d4_corr])

xtbmm = ash.QMMMTheory(
    fragment=frag, 
    qm_theory=qm_xtb,
    mm_theory=mm, 
    qmatoms=qm_atoms,
    embedding="elstat", 
    qm_charge=-1,
    qm_mult=1, 
    printlevel=2
)

dftmm = ash.QMMMTheory(
    fragment=frag, 
    qm_theory=qm_r2scan3c,
    mm_theory=mm, 
    qmatoms=qm_atoms,
    embedding="elstat", 
    qm_charge=-1,
    qm_mult=1, 
    printlevel=2
)

# --- run with gradient ---
xtbmm.run(
    current_coords=frag.coords, 
    elems=frag.elems,
    Grad=True
)

dftmm.run(
    current_coords=frag.coords, 
    elems=frag.elems,
    Grad=True
)

print(f"QM energy : {dftmm.QMenergy:.6f} Hartree")
print(f"QM gradient :\n{dftmm.QMgradient}")
print(f"QM gradient (projected):\n{dftmm.QM_PC_gradient[dftmm.qmatoms]}")



