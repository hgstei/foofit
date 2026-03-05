# foofit

X-ray reflectivity (XRR) fitting library using the Parratt recursion formalism and the refraction-corrected master formula.

## Installation

Requires <a href="https://git-scm.com/downloads" target="_blank">git</a> to be installed.

```bash
pip install git+https://github.com/hgstei/foofit.git
```

## Quick start

```python
from foofit import *

# use the bundled example dataset (polystyrene on silicon)
si_ps_xrr = example_data

params = Parameters()
params.add('numbLayers', value=1, vary=False)
params.add('wavelength', value=1.0, vary=False)
params.add('I0', value=1, vary=False)
params.add('bkg', value=0, vary=False)
params.add('pre_rho', value=0, vary=False)
params.add('pre_beta', value=0, vary=False)
params.add('layer0_dd', value=220, min=150, max=300, vary=True)
params.add('layer0_rho', value=0.25, min=0.1, max=0.4, vary=True)
params.add('layer0_sig', value=5, min=0.5, max=15, vary=True)
params.add('layer0_beta', value=0, vary=False)
params.add('sub_rho', value=0.71, vary=False)
params.add('sub_sig', value=4, min=0.5, max=10, vary=True)
params.add('sub_beta', value=0, vary=False)

# single fit
performFit(si_ps_xrr, params, fitFunc=xrr_parratt_fit,
           method='powell', qmin=0.0, qmax=0.4, plot=2,
           outputName='my_sample', weight=2)

# Monte Carlo error analysis (parallelized)
params2 = loadParams('my_sample_<timestamp>_fit.fitParams', lowLim=0.3, highLim=0.3)
performFit_mc(si_ps_xrr, params2, fitFunc=xrr_parratt_fit,
              method='powell', qmin=0.0, qmax=0.4,
              outputName='my_sample', weight=2, NN=250)

# corner plot of MC parameter distributions
analyze_mc('my_sample_<timestamp>_fit_mc.fitParams')
```

See `example.ipynb` for a full worked example and [foofit_manual.pdf](https://github.com/hgstei/foofit/blob/master/foofit_manual.pdf) for the full manual.

## Interactive plots in Jupyter

Uncomment the appropriate line at the top of your notebook:

```python
# %matplotlib widget
# (JupyterLab — requires: pip install ipympl)
# %matplotlib notebook
# (classic Jupyter Notebook)
```

Note: do not add comments on the same line as a `%matplotlib` magic command — IPython does not treat `#` as a comment in magic arguments.

## Parameters convention

| Parameter | Description |
|---|---|
| `numbLayers` | number of layers between ambient and substrate (int, fixed) |
| `wavelength` | X-ray wavelength in Å (fixed) |
| `I0`, `bkg` | intensity scale factor and background |
| `pre_rho`, `pre_beta` | ambient medium optical constants |
| `layer{n}_dd` | layer thickness in Å |
| `layer{n}_rho` | layer electron density in e/Å³ |
| `layer{n}_sig` | layer roughness in Å |
| `layer{n}_beta` | layer absorption constant |
| `sub_rho`, `sub_sig`, `sub_beta` | substrate parameters |

Layer index `n` starts at 0 (topmost layer, closest to ambient).

## Functions

| Function | Description |
|---|---|
| `xrr_parratt_calc(params, qq, doConv=0)` | Parratt recursion reflectivity |
| `xrr_master_refractionCorrected_calc(params, qq, doConv=0)` | Refraction-corrected master formula |
| `xrr_parratt_fit` / `xrr_master_refractionCorrected_fit` | Residual functions for lmfit |
| `xrr_eDens(params, zz)` | Electron density profile |
| `xrr_eDens_zeroRoughness(params, zz)` | Electron density profile (sharp interfaces) |
| `xrr_beta(params, zz)` | Absorption profile |
| `xrr_beta_zeroRoughness(params, zz)` | Absorption profile (sharp interfaces) |
| `performFit(...)` | Single fit with plotting and file output |
| `performFit_mc(...)` | Monte Carlo error analysis (parallelized with joblib) |
| `loadParams(filename, lowLim, highLim)` | Load saved `.fitParams` file into lmfit Parameters |
| `analyze_mc(file, bins)` | Corner plot of MC parameter distributions |
| `example_data` | Path to the bundled `si_ps.xrr` example dataset |

## Physical constants

Classical electron radius and critical-q prefactor are computed from astropy constants at import time:

```python
r_e_AA   # classical electron radius in Å  (from astropy sigma_T)
qc_factor  # 4 * sqrt(r_e_AA * pi)
```
