# punk The Period-UNKown initialization engine.

```punk``` is a Python package for extracting physical properties of Solar System Objects (SSOs) from sparse multi-band photometry, using the ```SOCCA``` model (Shape, Orientation, and Colors Combined Algorithm) (citation needed).

It is designed for large survey data (e.g. LSST, ZTF), where traditional lightcurve inversion is computationaly too costly.

### Features
- Joint modeling of:

1. Absolute magnitude ($H$)
2. Phase function ($G_1, G_2$)
3. Rotation period ($O_\text{sid}$)
4. Spin axis ($\alpha, \delta$)
5. Shape (triaxial ellipsoid: $a/b, a/c$)

- Works with sparse, irregularly sampled photometry
- Multi-band fitting
- Scalable to large datasets (survey-ready)
- Built-in:
1. Period search (Lomb–Scargle + model selection)
2. Alias rejection
3. Initialization via ```sHG1G2``` (citation needed)

### Installation
```
git clone https://github.com/OdysseasXenos/punk
cd punk
pip install -e .

```

### Quick example
```
from punk.rock import initialize 
import phunk

pc = phunk.PhaseCurve(
    target=target,
    epoch=data["Date"],
    phase=data["Phase"],
    mag=data["i:magpsf_red"],
    mag_err=data["i:sigmapsf"],
    band=data["i:fid"],
)
pc.get_ephems()

p0, metadata = initialize(pc, weights=pc.mag_err, remap=True, metadata=False)
pc.fit(models=["SOCCA"], p0=p0, weights=pc.mag_err, remap=True)
```

### How ```punk``` Works 
The fitting process is staged to avoid local minima:
- Initial Fit with ```sHG1G2```
- Provides:
1. $H, G_1, G_2$
2. Spin-axis parameter space
3. First guess for $a/b, a/c$

- Period Search
1. Lomb–Scargle on ```sHG1G2``` residuals
2. Harmonics and aliases flagging
3. Bootstrap stability test

- Full SOCCA Fit
