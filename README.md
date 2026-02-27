# foofit

X-ray reflectivity (XRR) fitting library using the Parratt recursion formalism and the refraction-corrected master formula.

## Installation

```bash
pip install -e .
```

## Usage

```python
from foofit import *
from lmfit import Parameters

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

# Single fit
performFit('data.xrr', params, fitFunc=xrr_parratt_fit,
           method='powell', qmin=0.0, qmax=0.4, plot=2,
           outputName='my_sample', weight=2)

# Monte Carlo error analysis (parallelized)
performFit_mc('data.xrr', params, fitFunc=xrr_parratt_fit,
              method='powell', qmin=0.0, qmax=0.4,
              outputName='my_sample', weight=2, NN=250)
```

## Functions

| Function | Description |
|---|---|
| `xrr_parratt_calc` | Parratt recursion reflectivity calculation |
| `xrr_master_refractionCorrected_calc` | Refraction-corrected master formula calculation |
| `xrr_parratt_fit` | Parratt fit residual (for lmfit Minimizer) |
| `xrr_master_refractionCorrected_fit` | Master formula fit residual |
| `xrr_eDens` | Electron density profile |
| `xrr_eDens_zeroRoughness` | Electron density profile (sharp interfaces) |
| `xrr_beta` | Absorption profile |
| `performFit` | Single fit with plotting and output |
| `performFit_mc` | Monte Carlo error analysis (parallelized) |
| `loadParams` | Load saved fit parameters |
| `analyze_mc` | Corner plot of MC parameter distributions |
