# punk 
## The Period-UNKown initialization engine.

```punk``` is a Python package for extracting physical properties of Solar System Objects (SSOs) from sparse multi-band photometry, 
using the ```SOCCA``` model (Shape, Orientation, and Colors Combined Algorithm) (citation needed).
It is designed for large survey data (e.g. LSST, ZTF).
It requires ```phunk``` 

### Features
- Joint modeling of:

1. Phase function ($H$, $G_1, G_2$)
2. Sidereal rotation period ($P_\text{sid}$)
3. Spin axis orientation ($\alpha, \delta$)
4. Shape (triaxial ellipsoid: $a/b, a/c$)

- Works with sparse, irregularly sampled photometry
- Multi-band fitting
- Scalable to large datasets (survey-ready)
- Built-in:
1. Period search (Lomb–Scargle + model selection)
2. Period alias and bogus flagging 
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
- Initial fit with ```sHG1G2``` which provides:
1. $H, G_1, G_2$
2. Spin axis parameter space
3. First guess for $a/b, a/c$

- Period search
1. Lomb–Scargle on ```sHG1G2``` residuals
2. Harmonics and aliases flagging
3. Bootstrap stability test for bogus flagging
4. Correction for the synodic-sideral period difference

- Full SOCCA fit on the spin axis minima and selected trial periods

### Photometric model
The observed apparent magnitude $m$ is modeled as:
$$m = H + f(r,\Delta) + g(\gamma) + s(\alpha, \delta, P_\text{sid}, a/b, a/c, W_0)$$

where:
- $f(r,\Delta)=5\log_{10}(r,\Delta)$ accounts for the object-observer distance variation
- $g(\gamma)$ is the phase function which accounts for the variations due to the Sun-object-observer angle ($\gamma$)
with 
$$g(\gamma) = -2.5\log_{10}(G_1\phi_1(\gamma) + G_2\phi_2(\gamma) + (1-G_1-G_2)\phi_3(\gamma))$$
and the function $s$
which models the photometric variation coming from the projection of a spinning triaxial ellipsoid.
