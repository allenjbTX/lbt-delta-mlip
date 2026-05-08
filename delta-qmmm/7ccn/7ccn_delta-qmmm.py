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

qm_atoms = get_qm_atoms_from_pdb(qm_region_pdb)
print(f"QM atoms: {qm_atoms}")

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



