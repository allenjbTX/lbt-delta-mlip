"""
Build a serialized OpenMM System XML and a topology PDB from the Amber
12-6-4 prmtop/inpcrd files.

The System XML carries the full force field including the 12-6-4 C4 terms
(as a CustomNonbondedForce, loaded via ParmEd). The topology PDB provides
atom names, residues, and connectivity that ASH's OpenMMTheory uses to map
atom indices.

These two files are the inputs needed by:

    OpenMMTheory(xmlsystemfile="system.xml", pdbfile="system.pdb", ...)

for QM/MM single points. They are produced once and reused across all
snapshots.
"""

import openmm
import openmm.app as app
import openmm.unit as u
import parmed as pmd

# ============================================================================
# User configuration
# ============================================================================
PRMTOP = "7ccn_solv_1264.prmtop"
INPCRD = "7ccn_solv_1264.inpcrd"

OUT_SYSTEM_XML = "system.xml"
OUT_TOPOLOGY_PDB = "system.pdb"

CUTOFF_NM = 1.0          # 10 A, matches the relaxation
USE_HBOND_CONSTRAINTS = False   
USE_RIGID_WATER = False

# ============================================================================
# Build the System through ParmEd (required for 12-6-4)
# ============================================================================
print(f"Loading {PRMTOP} + {INPCRD} via ParmEd...")
parm = pmd.load_file(PRMTOP, INPCRD)
print(f"  atoms: {len(parm.atoms)}")
print(f"  residues: {len(parm.residues)}")
print(f"  box: {parm.box}")

system = parm.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=CUTOFF_NM * u.nanometer,
    constraints=app.HBonds if USE_HBOND_CONSTRAINTS else None,
    rigidWater=USE_RIGID_WATER,
)

# ============================================================================
# Verify the 12-6-4 C4 terms are present
# ============================================================================
force_types = [type(f).__name__ for f in system.getForces()]
print("\nForces in System:")
for name in force_types:
    print(f"  {name}")

assert "CustomNonbondedForce" in force_types, (
    "No CustomNonbondedForce found -- the 12-6-4 C4 terms were NOT loaded. "
    "Check that ParmEd is up to date and the prmtop has a "
    "LENNARD_JONES_CCOEF section."
)
print("\n12-6-4 C4 terms confirmed present.")

# ============================================================================
# Make sure no barostat is in the System
# ============================================================================
# parm.createSystem() does not add a barostat by default, but check just in
# case anything has been added inadvertently. Single-point QM/MM evaluations
# don't need (and shouldn't have) a barostat in the System -- it doesn't
# contribute to the potential energy but it would activate during any MD
# you later drive through this OpenMMTheory.
for i in reversed(range(system.getNumForces())):
    if isinstance(system.getForce(i), openmm.MonteCarloBarostat):
        print(f"Removing MonteCarloBarostat at force index {i}.")
        system.removeForce(i)

# ============================================================================
# Serialize System and write topology PDB
# ============================================================================
with open(OUT_SYSTEM_XML, "w") as f:
    f.write(openmm.XmlSerializer.serialize(system))
print(f"\nWrote {OUT_SYSTEM_XML}")

parm.save(OUT_TOPOLOGY_PDB, overwrite=True)
print(f"Wrote {OUT_TOPOLOGY_PDB}")

# ============================================================================
# Sanity check: round-trip the System XML and re-verify
# ============================================================================
with open(OUT_SYSTEM_XML) as f:
    system_loaded = openmm.XmlSerializer.deserialize(f.read())
loaded_force_types = [type(f).__name__ for f in system_loaded.getForces()]
print("\nForces in deserialized System:")
for name in loaded_force_types:
    print(f"  {name}")
assert "CustomNonbondedForce" in loaded_force_types
print("\nRound-trip verified. Files are ready for ASH.")
