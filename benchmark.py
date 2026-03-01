"""Benchmark and correctness test for foofit speed optimizations."""
import numpy as np
import timeit
from lmfit import Parameters
from foofit import *

# --- Parameters matching example.ipynb (post-fit values) ---
params = Parameters()
params.add('numbLayers', value=1,     vary=False)
params.add('wavelength', value=1.0,   vary=False)
params.add('I0',         value=1.0,   vary=False)
params.add('bkg',        value=0.0,   vary=False)
params.add('pre_rho',    value=0.0,   vary=False)
params.add('pre_beta',   value=0.0,   vary=False)
params.add('layer0_dd',  value=222.0, vary=False)
params.add('layer0_rho', value=0.246, vary=False)
params.add('layer0_sig', value=5.3,   vary=False)
params.add('layer0_beta',value=0.0,   vary=False)
params.add('sub_rho',    value=0.71,  vary=False)
params.add('sub_sig',    value=4.0,   vary=False)
params.add('sub_beta',   value=0.0,   vary=False)

qq   = np.linspace(0.01, 0.5, 500)
zz   = np.linspace(-50, 350, 2000)
N_fast = 2000   # repeats for fast functions
N_conv = 200    # repeats for convolution (slower)

print("=" * 55)
print("TIMINGS")
print("=" * 55)

t = timeit.timeit(lambda: xrr_parratt_calc(params, qq, doConv=0), number=N_fast)
print(f"xrr_parratt_calc (no conv):      {t/N_fast*1000:7.3f} ms")

t = timeit.timeit(lambda: xrr_parratt_calc(params, qq, doConv=0.01), number=N_conv)
print(f"xrr_parratt_calc (with conv):    {t/N_conv*1000:7.3f} ms")

t = timeit.timeit(lambda: xrr_master_refractionCorrected_calc(params, qq, doConv=0), number=N_fast)
print(f"xrr_master_calc (no conv):       {t/N_fast*1000:7.3f} ms")

t = timeit.timeit(lambda: xrr_master_refractionCorrected_calc(params, qq, doConv=0.01), number=N_conv)
print(f"xrr_master_calc (with conv):     {t/N_conv*1000:7.3f} ms")

t = timeit.timeit(lambda: xrr_eDens(params, zz), number=N_fast)
print(f"xrr_eDens:                       {t/N_fast*1000:7.3f} ms")

t = timeit.timeit(lambda: xrr_beta(params, zz), number=N_fast)
print(f"xrr_beta:                        {t/N_fast*1000:7.3f} ms")

print()
print("=" * 55)
print("REFERENCE VALUES (for correctness check)")
print("=" * 55)
rr_p   = xrr_parratt_calc(params, qq, doConv=0)
rr_pc  = xrr_parratt_calc(params, qq, doConv=0.01)
rr_m   = xrr_master_refractionCorrected_calc(params, qq, doConv=0)
rr_mc  = xrr_master_refractionCorrected_calc(params, qq, doConv=0.01)
dens   = xrr_eDens(params, zz)
beta_z = xrr_beta(params, zz)

print(f"parratt (no conv)  checksum: {np.sum(rr_p):.10f}")
print(f"parratt (conv)     checksum: {np.sum(rr_pc):.10f}")
print(f"master  (no conv)  checksum: {np.sum(rr_m):.10f}")
print(f"master  (conv)     checksum: {np.sum(rr_mc):.10f}")
print(f"eDens              checksum: {np.sum(dens):.10f}")
print(f"beta               checksum: {np.sum(beta_z):.10f}")
