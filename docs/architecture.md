# Architecture

## Stages

| Stage | Input | Output | Source | State |
|---|---|---|---|---|
| Thermal | recipe s | T(x, t), dT/ds | external MFEM solver with Enzyme derivatives | external |
| Descriptors | thermal jet at each point | G, cooling rate, R, and d/ds | `solidification.py` | verified on an analytic source |
| Microstructure, forward reference | thermal events | grains, texture | cellular automaton, such as ExaCA | not connected |
| Microstructure, differentiable | descriptors | grain width, aspect ratio, texture measures, and d/ds | to build | open problem |
| Properties | voxel microstructure | directional yield, and d/d(texture) | Fierro EVPFFT | not connected |
| Inverse | target descriptors | recipe | `inverse.py` | verified on an analytic source |

## Contracts

**Thermal jet.** One record per material point and crossing, in SI units, in the build
frame: `T_t`, `T_tt`, `grad` (3), `grad_t` (3), `T_s` (n), `T_ts` (n), `grad_s` (3, n).
The parameter list and its order travel with the file. Exported derivatives are with
respect to log parameters unless the file says otherwise.

**Cooling events.** Every resolved melting and solidification cycle at a point is kept,
so remelting is not lost.

**Microstructure.** Voxel grid, orientation convention, spacing, origin and build frame
are written with the data, and tested with a single crystal of known orientation.

## Rules

- The discrete grain model stays the forward reference. Its grain identities jump, so
  derivatives go through continuous descriptors or ensemble averages, and are checked
  against perturbations of the forward model.
- Every derivative has a finite difference gate before a downstream stage may use it.
- Derivatives come from the residual at the converged state, not through solver
  iterations.
- Third-party solver source, unpublished results and measured data stay outside this
  repository, under `private/` or `data/`, which git ignores.
