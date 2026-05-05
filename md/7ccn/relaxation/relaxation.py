"""
OpenMM translation of the Amber relaxation protocol for a lanthanide-binding peptide
using the 12-6-4 nonbonded model.

Stages (mirroring 1min..10md from the Amber protocol):
  1)  1min      -- minimize solvent, peptide (resid 1-18) restrained @ 100 kcal/mol/A^2
  2)  2mdheat   -- NVT, heat 100 -> 298.15 K over 1 ns, peptide @ 100,  dt=1 fs
  3)  3md+3rs   -- NPT eq, 1 ns total, peptide @ 100, dt=1 fs
  4)  4md       -- NPT, 1 ns, peptide @ 10, dt=1 fs
  5)  5min      -- minimize, switch to backbone (CA/N/C) restraints @ 10
  6)  6md       -- NPT, 1 ns, backbone @ 10, dt=1 fs (FRESH velocities at 298.15 K)
  7)  7md       -- NPT, 1 ns, backbone @ 1,  dt=1 fs
  8)  8md       -- NPT, 1 ns, backbone @ 0.1, dt=1 fs
  9)  9md       -- NPT, 1 ns, no restraints, dt=1 fs
 10)  10md      -- NPT, 1 ns, no restraints, dt=2 fs
 11)  11md      -- NVT, 1 ns, no restraints, dt=2 fs

Notes:
  * Loads Amber 12-6-4 prmtop via ParmEd so the C4 terms become a
    CustomNonbondedForce in the OpenMM System (OpenMM's native AmberPrmtopFile
    reader silently drops them).
  * Position restraints are implemented as a CustomExternalForce with a global
    parameter `k` so we can change the force constant between stages without
    rebuilding the System. We rebuild the force when the *atom set* changes
    (peptide -> backbone -> none).
  * Amber's E = k_amber * r^2 convention is matched by using the OpenMM
    expression "k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)" -- no factor of 1/2.
  * State (positions + velocities + box) is checkpointed between stages via
    OpenMM XML state files.
"""

import openmm
import openmm.app as app
import openmm.unit as u
import parmed as pmd
import numpy as np
import os
import gc

# =============================================================================
# User configuration
# =============================================================================
PRMTOP = "7ccn_solv_1264.prmtop"
INPCRD = "7ccn_solv_1264.inpcrd"

PEPTIDE_RESID_RANGE = (1, 18)        # 1-indexed Amber residue range
BACKBONE_ATOM_NAMES = {"CA", "N", "C"}

PLATFORM = "CUDA"                    # 'CUDA', 'OpenCL', 'CPU', or 'Reference'
CUTOFF_NM = 1.0                      # 10 A
TEMPERATURE = 298.15 * u.kelvin
PRESSURE = 1.0 * u.bar
FRICTION = 1.0 / u.picosecond        # gamma_ln = 1.0 ps^-1
BAROSTAT_FREQ = 100                  # mcbarint = 100 (steps)

OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# Build the OpenMM System through ParmEd (required for 12-6-4)
# =============================================================================
print("Loading prmtop/inpcrd via ParmEd...")
parm = pmd.load_file(PRMTOP, INPCRD)

system = parm.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=CUTOFF_NM * u.nanometer,
    constraints=app.HBonds,          # SHAKE-equivalent for H bonds (ntc=2, ntf=2)
    rigidWater=True,
)

# Sanity check: confirm a CustomNonbondedForce is present (12-6-4 C4 terms).
force_types = [type(f).__name__ for f in system.getForces()]
print("Forces in System:", force_types)
assert "CustomNonbondedForce" in force_types, (
    "No CustomNonbondedForce found -- 12-6-4 C4 terms were NOT loaded. "
    "Make sure ParmEd is up to date and the prmtop has a LENNARD_JONES_CCOEF section."
)

# Add MC barostat for NPT stages. Set frequency to 0 for NVT stages by toggling
# the global parameter MonteCarloPressure.
barostat = openmm.MonteCarloBarostat(PRESSURE, TEMPERATURE, BAROSTAT_FREQ)
barostat_index = system.addForce(barostat)
# Disable initially (NVT for stage 1+2). We'll enable it for stages 3+.
# NB: simpler to toggle by setting the frequency to a huge number; we use a
#     global parameter approach instead.

# Identify atom indices we'll need:
peptide_atoms = []
backbone_atoms = []
for atom in parm.atoms:
    # ParmEd residue numbers are 1-indexed (matching Amber)
    if PEPTIDE_RESID_RANGE[0] <= atom.residue.idx + 1 <= PEPTIDE_RESID_RANGE[1]:
        peptide_atoms.append(atom.idx)
        if atom.name in BACKBONE_ATOM_NAMES:
            backbone_atoms.append(atom.idx)

print(f"Peptide atoms: {len(peptide_atoms)}  (residues "
      f"{PEPTIDE_RESID_RANGE[0]}-{PEPTIDE_RESID_RANGE[1]})")
print(f"Backbone atoms (CA/N/C): {len(backbone_atoms)}")


# =============================================================================
# Position-restraint helpers
# =============================================================================
# We use OpenMM's `periodicdistance` so the restraint behaves correctly with
# PBC. Energy = k * d^2 (matches Amber's restraint_wt convention).
RESTRAINT_EXPR = "k*periodicdistance(x, y, z, x0, y0, z0)^2"


def add_position_restraint(system, atom_indices, ref_positions_nm,
                           k_kcal_per_mol_A2):
    """Attach a position-restraint CustomExternalForce. Returns (force, idx)."""
    force = openmm.CustomExternalForce(RESTRAINT_EXPR)
    # Convert kcal/(mol*A^2) -> kJ/(mol*nm^2):  *4.184 (kcal->kJ) *100 (A^-2->nm^-2)
    k_omm = k_kcal_per_mol_A2 * 4.184 * 100.0
    force.addGlobalParameter("k", k_omm * u.kilojoule_per_mole / u.nanometer**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i, idx in enumerate(atom_indices):
        x0, y0, z0 = ref_positions_nm[i]
        force.addParticle(int(idx), [float(x0), float(y0), float(z0)])
    f_idx = system.addForce(force)
    return force, f_idx


def remove_force(system, f_idx):
    system.removeForce(f_idx)


# =============================================================================
# Simulation factory
# =============================================================================
def make_simulation(timestep_fs, temperature):
    """Build a fresh Simulation. Required when we change timestep or after
    we mutate the System forces."""
    integrator = openmm.LangevinMiddleIntegrator(
        temperature, FRICTION, timestep_fs * u.femtosecond
    )
    platform = openmm.Platform.getPlatformByName(PLATFORM)
    sim = app.Simulation(parm.topology, system, integrator, platform)
    return sim


def load_state(sim, state_path):
    sim.loadState(state_path)


def save_state(sim, state_path):
    sim.saveState(state_path)


def set_initial_positions_and_box(sim):
    sim.context.setPositions(parm.positions)
    if parm.box_vectors is not None:
        sim.context.setPeriodicBoxVectors(*parm.box_vectors)


def get_current_positions_nm(sim, atom_indices):
    state = sim.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    return [pos[i] for i in atom_indices]


def attach_reporters(sim, base, traj_every, log_every, total_steps,
                     append=False):
    sim.reporters.clear()
    sim.reporters.append(app.DCDReporter(
        os.path.join(OUTDIR, f"{base}.dcd"), traj_every, append=append
    ))
    sim.reporters.append(app.StateDataReporter(
        os.path.join(OUTDIR, f"{base}.log"), log_every,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        totalEnergy=True, temperature=True, volume=True, density=True,
        speed=True, totalSteps=total_steps, separator="\t",
    ))

def load_state_partial(sim, state_path):
    """Load positions, velocities, and box from an OpenMM state XML,
    ignoring any global parameters that may not exist in the current System."""
    with open(state_path) as f:
        state = openmm.XmlSerializer.deserialize(f.read())
    sim.context.setPositions(state.getPositions())
    sim.context.setVelocities(state.getVelocities())
    sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())

def release(sim):
    """Force-release a Simulation's CUDA context before creating the next one."""
    if sim is not None:
        # Drop reporters first (they may hold open file handles / state refs)
        sim.reporters.clear()
        del sim.context
        del sim.integrator
    gc.collect()


# =============================================================================
# Stage 1: 1min -- minimize solvent, peptide restrained @ 100 kcal/mol/A^2
# =============================================================================
print("\n=== Stage 1: 1min (solvent minimization, peptide restraint 100) ===")

# Reference positions for the peptide come from the input coordinates.
ref_positions_nm = np.array(
    parm.positions.value_in_unit(u.nanometer)
)
peptide_ref = [ref_positions_nm[i] for i in peptide_atoms]

restraint_force, restraint_idx = add_position_restraint(
    system, peptide_atoms, peptide_ref, k_kcal_per_mol_A2=100.0
)

sim = make_simulation(timestep_fs=1.0, temperature=TEMPERATURE)
set_initial_positions_and_box(sim)

sim.minimizeEnergy(tolerance=10 * u.kilojoule_per_mole / u.nanometer,
                   maxIterations=1000)
save_state(sim, os.path.join(OUTDIR, "1min.xml"))

# =============================================================================
# Stage 2: 2mdheat -- NVT heating 100 -> 298.15 K over 1 ns @ 1 fs
# =============================================================================
print("\n=== Stage 2: 2mdheat (NVT, 100 -> 298.15 K, 1 ns, peptide @ 100) ===")

# Already have peptide restraint @ 100; just rebuild simulation with new T_init.
release(sim)

# Disable barostat for NVT.
barostat.setFrequency(0)

sim = make_simulation(timestep_fs=1.0, temperature=100.0 * u.kelvin)
load_state(sim, os.path.join(OUTDIR, "1min.xml"))
sim.context.setVelocitiesToTemperature(100.0 * u.kelvin)

heat_total_steps = 1_000_000
chunk_steps = 10_000          # update T every 10 ps
n_chunks = heat_total_steps // chunk_steps
attach_reporters(sim, "2mdheat", traj_every=10000, log_every=1000,
                 total_steps=heat_total_steps)

for i in range(n_chunks):
    frac = (i + 0.5) / n_chunks    # midpoint of this chunk
    T = (100.0 + frac * (298.15 - 100.0)) * u.kelvin
    sim.integrator.setTemperature(T)
    sim.step(chunk_steps)

save_state(sim, os.path.join(OUTDIR, "2mdheat.xml"))

# =============================================================================
# Stage 3: 3md (+ restart) -- NPT, 1 ns total, peptide @ 100, dt=1 fs
# =============================================================================
print("\n=== Stage 3: 3md (NPT eq, 1 ns, peptide @ 100) ===")

barostat.setFrequency(BAROSTAT_FREQ)
# Restraint already at k=100; no change needed.

release(sim)
sim = make_simulation(timestep_fs=1.0, temperature=TEMPERATURE)
load_state(sim, os.path.join(OUTDIR, "2mdheat.xml"))
attach_reporters(sim, "3md", traj_every=10000, log_every=1000,
                 total_steps=1_000_000)
sim.step(1_000_000)
save_state(sim, os.path.join(OUTDIR, "3md.xml"))

# =============================================================================
# Stage 4: 4md -- NPT, 1 ns, peptide @ 10
# =============================================================================
print("\n=== Stage 4: 4md (NPT, 1 ns, peptide @ 10) ===")

# Lower the restraint force constant via the global parameter.
sim.context.setParameter(
    "k", 10.0 * 4.184 * 100.0 * u.kilojoule_per_mole / u.nanometer**2
)
attach_reporters(sim, "4md", traj_every=10000, log_every=1000,
                 total_steps=1_000_000)
sim.step(1_000_000)
save_state(sim, os.path.join(OUTDIR, "4md.xml"))

# =============================================================================
# Stage 5: 5min -- swap to backbone restraints @ 10, then minimize
# =============================================================================
print("\n=== Stage 5: 5min (backbone @ 10, minimization) ===")

# Get current positions to use as new reference for the backbone restraints.
backbone_ref = get_current_positions_nm(sim, backbone_atoms)

# Remove the peptide-wide restraint, add backbone-only restraint.
remove_force(system, restraint_idx)
restraint_force, restraint_idx = add_position_restraint(
    system, backbone_atoms, backbone_ref, k_kcal_per_mol_A2=10.0
)

# Rebuild simulation because we mutated the System (added/removed forces).
release(sim)
sim = make_simulation(timestep_fs=1.0, temperature=TEMPERATURE)
load_state(sim, os.path.join(OUTDIR, "4md.xml"))   # positions/velocities/box

sim.minimizeEnergy(tolerance=10 * u.kilojoule_per_mole / u.nanometer,
                   maxIterations=1000)
save_state(sim, os.path.join(OUTDIR, "5min.xml"))

# =============================================================================
# Stage 6: 6md -- NPT, 1 ns, backbone @ 10, FRESH velocities at 298.15 K
# =============================================================================
print("\n=== Stage 6: 6md (NPT, 1 ns, backbone @ 10, fresh velocities) ===")

release(sim)
sim = make_simulation(timestep_fs=1.0, temperature=TEMPERATURE)
load_state(sim, os.path.join(OUTDIR, "5min.xml"))
# Mirror Amber's ntx=1, irest=0 -- assign new velocities at target T.
sim.context.setVelocitiesToTemperature(TEMPERATURE)

attach_reporters(sim, "6md", traj_every=10000, log_every=1000,
                 total_steps=1_000_000)
sim.step(1_000_000)
save_state(sim, os.path.join(OUTDIR, "6md.xml"))

# =============================================================================
# Stage 7: 7md -- backbone @ 1, 1 ns
# =============================================================================
print("\n=== Stage 7: 7md (NPT, 1 ns, backbone @ 1) ===")
sim.context.setParameter(
    "k", 1.0 * 4.184 * 100.0 * u.kilojoule_per_mole / u.nanometer**2
)
attach_reporters(sim, "7md", traj_every=10000, log_every=1000,
                 total_steps=1_000_000)
sim.step(1_000_000)
save_state(sim, os.path.join(OUTDIR, "7md.xml"))

# =============================================================================
# Stage 8: 8md -- backbone @ 0.1, 1 ns
# =============================================================================
print("\n=== Stage 8: 8md (NPT, 1 ns, backbone @ 0.1) ===")
sim.context.setParameter(
    "k", 0.1 * 4.184 * 100.0 * u.kilojoule_per_mole / u.nanometer**2
)
attach_reporters(sim, "8md", traj_every=10000, log_every=1000,
                 total_steps=1_000_000)
sim.step(1_000_000)
save_state(sim, os.path.join(OUTDIR, "8md.xml"))

# =============================================================================
# Stage 9: 9md -- no restraints, 1 ns
# =============================================================================
print("\n=== Stage 9: 9md (NPT, 1 ns, no restraints) ===")
remove_force(system, restraint_idx)
restraint_idx = None

release(sim)
sim = make_simulation(timestep_fs=1.0, temperature=TEMPERATURE)
load_state_partial(sim, os.path.join(OUTDIR, "8md.xml"))
attach_reporters(sim, "9md", traj_every=10000, log_every=1000,
                 total_steps=1_000_000)
sim.step(1_000_000)
save_state(sim, os.path.join(OUTDIR, "9md.xml"))

# =============================================================================
# Stage 10: 10md -- 2 fs timestep, 1 ns, no restraints
# =============================================================================
print("\n=== Stage 10: 10md (NPT, 1 ns, dt=2 fs, no restraints) ===")

# Need a new integrator with dt=2 fs.
release(sim)
sim = make_simulation(timestep_fs=2.0, temperature=TEMPERATURE)
load_state(sim, os.path.join(OUTDIR, "9md.xml"))
# 1 ns @ 2 fs = 500,000 steps
attach_reporters(sim, "10md", traj_every=5000, log_every=500,
                 total_steps=500_000)
sim.step(500_000)
save_state(sim, os.path.join(OUTDIR, "10md.xml"))

# =============================================================================
# Stage 11: 11md -- 2 fs timestep, 1 ns, no restraints, NVT ensemble
# =============================================================================
print("\n=== Stage 11: 11md (NVT, 1 ns, dt=2 fs, no restraints) ===")

# Need a new integrator with dt=2 fs.
release(sim)

# Disable barostat for NVT.
barostat.setFrequency(0)

sim = make_simulation(timestep_fs=2.0, temperature=TEMPERATURE)
load_state(sim, os.path.join(OUTDIR, "10md.xml"))
# 1 ns @ 2 fs = 500,000 steps
attach_reporters(sim, "11md", traj_every=5000, log_every=500,
                 total_steps=500_000)
sim.step(500_000)
save_state(sim, os.path.join(OUTDIR, "11md.xml"))

print("\nRelaxation protocol complete.")
print(f"Outputs in: {OUTDIR}/")
