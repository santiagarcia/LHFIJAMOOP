# LHFIJAMOOP

Low Hanging Fruit Is Just A Matter Of Outer Perception.

A derivative-based framework that goes from laser printing parameters to a desired
microstructure, and from there to the mechanical response.

```
printing parameters  ->  thermal history  ->  solidification descriptors  ->  microstructure  ->  yield
       s                    T, dT/ds              G, R, Tdot + d/ds            grains, texture     Fierro
```

Each stage returns its value and its derivative with respect to the printing parameters,
so the chain can be run backward to find a recipe for a target microstructure.

## Status

Early. What exists today:

- `solidification`: thermal gradient, cooling rate and front speed at the liquidus
  crossing, with total parameter derivatives from the implicit function theorem.
- `rosenthal`: analytic moving source used to verify the chain.
- `inverse`: bounded Gauss-Newton recipe solve.
- `recipe`: first-order prediction of melt quantities across recipes.

The thermal solver and the crystal plasticity solver are separate projects. They connect
through the file contracts in [docs/architecture.md](docs/architecture.md), and neither
is included here.

## Use

```
pip install -e .
python -m unittest discover -s tests
```
