# Gas–Dust Dynamics Model via Interpolated Hydrodynamic Snapshots

This repository contains a prototype model for simulating gas–dust dynamics by interpolating precomputed hydrodynamic snapshots and evolving dust particle trajectories under gravity and gas drag. The primary goal is to explore particle dynamics efficiently without re-running full gas simulations.
 *Note: This code is under active development and should be considered experimental.*

 Contact the author for the hydrodynamic data or access it from [Tārā](https://plg.physlab.uni-due.de/tara-database/)

 # Overview
 - Uses interpolated gas velocity and density fields from hydrodynamic snapshots
 - Evolves dust particles under gravity and gas drag
 - Is intended for studying dust settling and vertical dynamics in disk-like systems

# Current Limitations and Known Issues
1. Midplane Crossing Instabilities
   The model exhibits unphysical behaviour when particles cross the disk midplane. This issue may originate from:
     - Interpolation artefacts in the hydrodynamic snapshots, or
     - The current symmetry mapping that folds particle positions into the first quadrant.
    Further testing is required to determine whether the instability is numerical (interpolation-related) or geometric (coordinate mapping–related).

2. Unit Sensitivity of the Stopping Time
   The model currently behaves correctly only when the stopping time is specified in seconds, and fails when expressed in years. This strongly suggests an unresolved unit-conversion inconsistency in the gas drag implementation.
   Resolving this issue is necessary before the model can be reliably used with physically meaningful parameter choices.

3. Validation Using Free-Fall Trajectories (Zero Drag Limit)
   To better understand the differences between this model and Tara’s dataset, the system should be tested in the zero gas-drag limit. In this regime, particle trajectories should follow analytic free-fall solutions. This test will help isolate errors arising from interpolation versus those inherent to the dynamical setup.

4. Time-Stepping Near the Stopping Time
   The model has not yet been systematically tested with:
     - Multiple time-step choices, or
     - Adaptive time stepping,
   particularly in regimes where the integration time step becomes comparable to the particle stopping time. Such regimes are expected to be numerically stiff, and dedicated testing is required to assess stability and accuracy.   
