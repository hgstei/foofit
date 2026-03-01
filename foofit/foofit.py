#title           : foofit.py
#description     : functions for calculating and fits reflectivity data with parratt formalism or refraction corrected master formula
#                  code based on pyRefFit by Michael Klimczak and Hans-Georg Steinrück
#author          : Hans-Georg Steinrück
#affiliation     : Institute for a sustainable Hydrogen Economy (IHE-1), Forschungszentrum Jülich &
#                  Institute of Physical Chemistry, RWTH Aachen University
#e-mail          : h.steinrueck@fz-juelich.de
#date            : 08/18/2018 (updated 02/2026)
#version         : 0.2.4
#usage           : best to use with jupyter notebook
#notes           : I would call this a beta version, needs to be tested by colleagues
#                  The code was cross-checked with genx
#python_version  : written for 3.5.4, tested on 3.7+

### to-do: potentially export fit-report to output file
### to-do: work on emcee
### to-do: add best fit to mc
### to-do: add multilayer option via a lmfit Parameter generation script
## document steps in formulas
###

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.special
import scipy.ndimage
import datetime
import time

from lmfit import Minimizer, Parameters, report_fit, fit_report

from prettytable import PrettyTable
from tqdm import tqdm

import corner

from joblib import Parallel, delayed

from astropy.constants import sigma_T

# Classical electron radius in Angstroms, derived from Thomson cross-section via astropy:
#   sigma_T = (8*pi/3) * r_e^2  =>  r_e = sqrt(3*sigma_T / (8*pi))
r_e_AA = np.sqrt(3 * sigma_T.si.value / (8 * np.pi)) * 1e10
# Critical q prefactor: qc = qc_factor * sqrt(delta_rho)  [rho in e/Å³, qc in Å⁻¹]
qc_factor = 4 * np.sqrt(r_e_AA * np.pi)

# Path to the bundled example data file
example_data = os.path.join(os.path.dirname(__file__), 'si_ps.xrr')


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def smear_scipy_int(qq, rr, sig):
    '''
    Convolves rr(qq) with a Gaussian of width sig (sigma in q units).
    Works for arbitrarily spaced qq by oversampling to a uniform fine grid,
    convolving, then interpolating back onto the original grid.
    '''
    dq = np.diff(qq)
    dq_fine = np.min(dq[dq != 0]) / 10
    qq_fine = np.arange(np.min(qq), np.max(qq) + dq_fine, dq_fine)
    rr_fine = np.interp(qq_fine, qq, rr)
    rr_fine_smeared = scipy.ndimage.gaussian_filter1d(rr_fine, sig / dq_fine)
    return np.interp(qq, qq_fine, rr_fine_smeared)


def _fresnel_rrf(qq, params):
    """Fresnel reflectivity normalization factor R_F for rrfPlot."""
    qc = qc_factor * np.sqrt(params['sub_rho'].value - params['pre_rho'].value)
    qq_p = np.sqrt(qq**2 - qc**2 + 0j)
    return np.abs((qq - qq_p) / (qq + qq_p))**2


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_parratt_calc(params, qq, doConv=0):

    I0 = params['I0'].value
    bkg = params['bkg'].value

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    rho = np.array([params['pre_rho'].value] +
                   [params[f'layer{nn}_rho'].value for nn in range(numbLayers)] +
                   [params['sub_rho'].value])

    beta = np.array([params['pre_beta'].value] +
                    [params[f'layer{nn}_beta'].value for nn in range(numbLayers)] +
                    [params['sub_beta'].value])

    sig = np.array([params[f'layer{nn}_sig'].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])

    dd = np.array([0] + [params[f'layer{nn}_dd'].value for nn in range(numbLayers)])

    numInterfaces = numbLayers + 1

    # --> this makes the wrong critical angle correct, gets relative change in rho and beta right
    beta -= beta[0]
    rho -= rho[0]

    # Fixed: use params['wavelength'].value (was hardcoded to 1.033)
    wavelength = params['wavelength'].value
    # Optimized: broadcasting instead of meshgrid (avoids two intermediate arrays)
    kz_sq = 8 * 2 * np.pi * r_e_AA * rho - 1j * 32 * np.pi**2 * beta * 1e-8 / wavelength**2
    qs = np.sqrt(qq**2 - kz_sq[:, np.newaxis])

    # Optimized: vectorized over interfaces (was a Python loop)
    # refraction-corrected roughness: qs[ii]*qs[ii+1] instead of qs[ii]**2
    r = (qs[:-1] - qs[1:]) / (qs[:-1] + qs[1:]) * np.exp(-sig[:, np.newaxis]**2 * qs[:-1] * qs[1:] / 2)
    p = np.exp(1j * qs[:-1] * dd[:, np.newaxis])

    # recursively build the reflective index of the entire system from the bottom up
    rr = r[numInterfaces-1]

    for ii in range(numInterfaces-2, -1, -1):
        rr = (r[ii] + rr * p[ii+1]) / (1 + r[ii] * rr * p[ii+1])

    rr = I0 * np.abs(rr)**2 + bkg

    if doConv != 0:
        rr = smear_scipy_int(qq, rr, doConv)

    return rr


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_master_refractionCorrected_calc(params, qq, doConv=0):

    I0 = params['I0'].value
    bkg = params['bkg'].value

    pre_rho = params['pre_rho'].value

    sub_rho = params['sub_rho'].value
    sub_sig = params['sub_sig'].value

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions + Fixed: was sig[nn-1] (off-by-one bug)
    rho = np.array([params[f'layer{nn}_rho'].value for nn in range(numbLayers)])
    dd = np.array([params[f'layer{nn}_dd'].value for nn in range(numbLayers)])
    sig = np.array([params[f'layer{nn}_sig'].value for nn in range(numbLayers)])

    layers = [(pre_rho, 0, 0)]
    for nn in range(numbLayers):
        layers.append((rho[nn], dd[nn], sig[nn]))
    layers.append((sub_rho, 0, sub_sig))

    qc = qc_factor * np.sqrt(sub_rho - pre_rho)

    qq_p = np.sqrt(qq**2 - qc**2 + 0j)   # complex: handles below-critical-angle (evanescent) region
    rrf = np.abs((qq - qq_p) / (qq + qq_p))**2

    # Vectorized over interfaces: compute all terms simultaneously
    n_int = len(layers) - 1
    rho_l    = np.array([layers[nn][0]   for nn in range(n_int)])
    rho_next = np.array([layers[nn+1][0] for nn in range(n_int)])
    dd_l     = np.array([layers[nn][1]   for nn in range(n_int)])
    sig_l    = np.array([layers[nn+1][2] for nn in range(n_int)])

    depths    = np.cumsum(dd_l)                                    # (n_int,)
    delta_rho = rho_l - rho_next                                   # (n_int,)
    phase     = np.exp(1j * np.outer(depths, qq_p))                # (n_int, len(qq))
    rough     = np.exp(-0.5 * np.outer(sig_l**2, qq**2))          # (n_int, len(qq))
    rr        = np.dot(delta_rho, phase * rough) / (sub_rho - pre_rho)

    rr = I0 * np.abs(rr)**2 * rrf + bkg

    if doConv != 0:
        rr = smear_scipy_int(qq, rr, doConv)

    return rr


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_parratt_fit(params, qq, data, weight, ee, doConv=0):
    # Optimized: delegate to _calc instead of duplicating physics code
    rr = xrr_parratt_calc(params, qq, doConv=doConv)

    if weight == 0:        # normalized
        res = (rr - data) / data
    elif weight == 1:      # linear
        res = (rr - data)
    elif weight == 2:      # log
        res = np.log(rr) - np.log(data)
    elif weight == 4:      # error-weighted
        res = (rr - data) / ee

    return res


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_master_refractionCorrected_fit(params, qq, data, weight, ee, doConv=0):
    # Optimized: delegate to _calc instead of duplicating physics code
    rr = xrr_master_refractionCorrected_calc(params, qq, doConv=doConv)

    if weight == 0:        # normalized
        res = (rr - data) / data
    elif weight == 1:      # linear
        res = (rr - data)
    elif weight == 2:      # log
        res = np.log(rr) - np.log(data)
    elif weight == 4:      # error-weighted
        res = (rr - data) / ee

    return res


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_eDens(params, zz, zero_roughness=False):

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    rho = np.array([params['pre_rho'].value] +
                   [params[f'layer{nn}_rho'].value for nn in range(numbLayers)] +
                   [params['sub_rho'].value])

    sig = np.array([params[f'layer{nn}_sig'].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])

    if zero_roughness:
        sig = np.full_like(sig, 0.0001)

    dd = np.array([0] + [params[f'layer{nn}_dd'].value for nn in range(numbLayers)])

    ZZ = np.cumsum(dd)

    # Vectorized over interfaces
    delta_rho = rho[1:] - rho[:-1]                                             # (numInterfaces,)
    z_norm = (zz[:, np.newaxis] - ZZ[np.newaxis, :]) / (np.sqrt(2) * sig)     # (len(zz), numInterfaces)
    density = rho[0] + np.sum(delta_rho * (1 + scipy.special.erf(z_norm)) / 2, axis=1)

    return density


def xrr_eDens_zeroRoughness(params, zz):
    return xrr_eDens(params, zz, zero_roughness=True)


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_beta(params, zz, zero_roughness=False):

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    beta = np.array([params['pre_beta'].value] +
                    [params[f'layer{nn}_beta'].value for nn in range(numbLayers)] +
                    [params['sub_beta'].value])

    sig = np.array([params[f'layer{nn}_sig'].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])

    if zero_roughness:
        sig = np.full_like(sig, 0.0001)

    dd = np.array([0] + [params[f'layer{nn}_dd'].value for nn in range(numbLayers)])

    ZZ = np.cumsum(dd)

    # Vectorized over interfaces
    delta_beta = beta[1:] - beta[:-1]
    z_norm = (zz[:, np.newaxis] - ZZ[np.newaxis, :]) / (np.sqrt(2) * sig)
    absorption = (beta[0] + np.sum(delta_beta * (1 + scipy.special.erf(z_norm)) / 2, axis=1)) * 1e-8

    return absorption


def xrr_beta_zeroRoughness(params, zz):
    return xrr_beta(params, zz, zero_roughness=True)


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def performFit(dataFile, params, fitFunc=xrr_parratt_fit, method='differential_evolution',
               qmin=0, qmax=1, plot=1, report=True, save=True, outputName="foo", weight=0, rrfPlot=False,
               doConv=0):

    plt.close('all')

    data = np.loadtxt(dataFile)

    qq = data[:,0]
    ii = data[:,1]
    try:
        ee = data[:,2]
        ebar = 1
    except (IndexError, ValueError):
        ee = np.zeros(len(ii))
        ebar = 0

    qq_cut = qq[(qq > qmin) & (qq < qmax)]
    ii_cut = ii[(qq > qmin) & (qq < qmax)]
    ee_cut = ee[(qq > qmin) & (qq < qmax)]

    start = time.perf_counter()

    minner = Minimizer(fitFunc, params, fcn_args=(qq_cut, ii_cut, weight, ee_cut, doConv), nan_policy='omit')
    result = minner.minimize(method=method)

    stop = time.perf_counter()
    print(f"time for fit: {stop - start:.2f} s")

    # calculate final result
    if fitFunc == xrr_parratt_fit:
        final = xrr_parratt_calc(result.params, qq_cut, doConv=doConv)
    elif fitFunc == xrr_master_refractionCorrected_fit:
        final = xrr_master_refractionCorrected_calc(result.params, qq_cut, doConv=doConv)

    qq_plot = np.arange(0, np.max(qq), 0.001)
    if fitFunc == xrr_parratt_fit:
        ii_plot = xrr_parratt_calc(result.params, qq_plot, doConv=doConv)
    elif fitFunc == xrr_master_refractionCorrected_fit:
        ii_plot = xrr_master_refractionCorrected_calc(result.params, qq_plot, doConv=doConv)

    # write error report
    if report:
        report_fit(result)

    if plot == 1:

        LBLU = "#ccf2ff"; YEL = "#f5f794"
        plt.rc("font", size=14); plt.rc('legend', **{'fontsize': 14}); plt.rcParams['font.family'] = 'M+ 2c'
        plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; plt.rcParams['figure.figsize'] = 9.3, 6
        plt.rcParams['axes.edgecolor'] = 'r'
        plt.rcParams["figure.facecolor"] = YEL
        fig = plt.figure(facecolor=YEL)

        gs = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])

        for ax in fig.get_axes():
            ax.tick_params(which='both', color='r')
            ax.set_facecolor(LBLU)

        ############################## Make plot for reflectivity
        ax1.set_xlabel("q$_\mathregular{z}$ (\u00c5$\mathregular{^{-1}}$)")
        ax1.set_ylabel("R/R$_\mathregular{F}$")
        ax1.yaxis.set_ticks_position('both'); ax1.xaxis.set_ticks_position('both')

        if rrfPlot:
            rrf = _fresnel_rrf(qq, result.params)
            ax1.semilogy(qq, ii/rrf, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
            if ebar == 1:
                ax1.errorbar(qq, ii/rrf, yerr=ee/rrf, linestyle='None', color='b', capsize=0, elinewidth=1)
            ax1.semilogy(qq_plot, ii_plot/_fresnel_rrf(qq_plot, result.params), color='k', linewidth=1)
            ax1.semilogy(qq_cut, final/_fresnel_rrf(qq_cut, result.params), color='r', linewidth=1)
        else:
            ax1.semilogy(qq, ii, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
            if ebar == 1:
                ax1.errorbar(qq, ii, yerr=ee, linestyle='None', color='b', capsize=0, elinewidth=1)
            ax1.semilogy(qq_plot, ii_plot, color='k', linewidth=1)
            ax1.semilogy(qq_cut, final, color='r', linewidth=1)

    if plot == 2:

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H.%M.%S')

        bestFitParam = []
        bestFitParam_name = []
        bestFitVary = []
        for item in result.params.items():
            bestFitParam.append(str(result.params[(item[0])].value))
            bestFitParam_name.append(str(item[0]))
            bestFitVary.append(str(result.params[(item[0])].vary))

        xrrOutput = np.column_stack((qq_plot, ii_plot))
        np.savetxt(f"{outputName}_{st}_fit.r", xrrOutput)

        paramOutput = np.column_stack((bestFitParam, bestFitParam_name, bestFitVary))
        np.savetxt(f"{outputName}_{st}_fit.fitParams", paramOutput, fmt="%s")

        LBLU = "#ccf2ff"; YEL = "#f5f794"
        plt.rc("font", size=14); plt.rc('legend', **{'fontsize': 14}); plt.rcParams['font.family'] = 'M+ 2c'
        plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; plt.rcParams['figure.figsize'] = 9.3, 6
        plt.rcParams['axes.edgecolor'] = 'r'
        plt.rcParams["figure.facecolor"] = YEL
        fig = plt.figure(facecolor=YEL)

        fig.suptitle(f"data file: {dataFile}", fontsize=14, y=1.0025)

        gs = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[:, 1])

        ############################## Make plot for reflectivity
        ax1.set_xlabel("q$_\mathregular{z}$ (\u00c5$\mathregular{^{-1}}$)")
        ax1.set_ylabel("R/R$_\mathregular{F}$")
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position("top")
        ax1.yaxis.set_ticks_position('both'); ax1.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both'); ax2.xaxis.set_ticks_position('both')

        for ax in fig.get_axes():
            ax.tick_params(which='both', color='r')
            ax.set_facecolor(LBLU)

        if rrfPlot:
            rrf = _fresnel_rrf(qq, result.params)
            ax1.semilogy(qq, ii/rrf, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
            if ebar == 1:
                ax1.errorbar(qq, ii/rrf, yerr=ee/rrf, linestyle='None', color='b', capsize=0, elinewidth=1)
            ax1.semilogy(qq_plot, ii_plot/_fresnel_rrf(qq_plot, result.params), color='k', linewidth=1)
            ax1.semilogy(qq_cut, final/_fresnel_rrf(qq_cut, result.params), color='r', linewidth=1)
        else:
            ax1.semilogy(qq, ii, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
            if ebar == 1:
                ax1.errorbar(qq, ii, yerr=ee, linestyle='None', color='b', capsize=0, elinewidth=1)
            ax1.semilogy(qq_plot, ii_plot, color='k', linewidth=1)
            ax1.semilogy(qq_cut, final, color='r', linewidth=1)

        ax1.annotate(f"q_min = {qmin}\nq_max = {qmax}", xy=(0.8, 0.8), xycoords='axes fraction', fontsize=8)

        ax2.set_ylabel("$\mathregular{\\rho}$ (e/\u00c5\u00b3)")
        ax2.set_xlabel("z (\u00c5)")

        DD = 0
        for nn in range(int(result.params['numbLayers'].value)):
            DD += result.params[f'layer{nn}_dd'].value
            topLayer_RR = result.params[f'layer{nn}_sig'].value
        sub_RR = result.params['sub_sig'].value
        zz = np.arange(0-7*topLayer_RR, DD+7*sub_RR, 0.01)

        dens = xrr_eDens(result.params, zz)
        dens_zeroRoughness = xrr_eDens_zeroRoughness(result.params, zz)

        edOutput = np.column_stack((zz, dens))
        np.savetxt(f"{outputName}_{st}_fit.ed", edOutput)

        nedOutput = np.column_stack((zz, dens_zeroRoughness))
        np.savetxt(f"{outputName}_{st}_fit.ned", nedOutput)

        if fitFunc == xrr_parratt_fit:
            beta_z = xrr_beta(result.params, zz)
            beta_zeroRoughness = xrr_beta_zeroRoughness(result.params, zz)

            betaOutput = np.column_stack((zz, beta_z))
            np.savetxt(f"{outputName}_{st}_fit.beta", betaOutput)

            nbetaOutput = np.column_stack((zz, beta_zeroRoughness))
            np.savetxt(f"{outputName}_{st}_fit.nbeta", nbetaOutput)

        ax2.plot(zz, dens, color='k', linewidth=1)
        ax2.plot(zz, dens_zeroRoughness, color='k', linewidth=1, linestyle='dashed')

        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.text(0.01, 0.01, fit_report(result), fontsize=8)

        gs.tight_layout(fig)

        if save:
            plt.savefig(f"{outputName}_{st}_fit.png", bbox_inches='tight', facecolor=YEL, dpi=600)


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def _single_mc_fit(params, qq_cut, ii_cut_nn, ee_cut, fitFunc, weight, doConv, method):
    """Run a single Monte Carlo fit. Defined at module level for joblib pickling."""
    minner = Minimizer(fitFunc, params, fcn_args=(qq_cut, ii_cut_nn, weight, ee_cut, doConv), nan_policy='omit')
    return minner.minimize(method=method)


def performFit_mc(dataFile, params, fitFunc=xrr_parratt_fit, method='differential_evolution',
                  qmin=0, qmax=1, plot=1, report=True,
                  outputName="foo", weight=0, rrfPlot=False, doConv=0, NN=10):

    plt.close('all')

    data = np.loadtxt(dataFile)

    qq = data[:,0]
    ii = data[:,1]
    try:
        ee = data[:,2]
        ebar = 1
    except (IndexError, ValueError):
        ee = np.zeros(len(ii))
        ebar = 0

    qq_cut = qq[(qq > qmin) & (qq < qmax)]
    ii_cut = ii[(qq > qmin) & (qq < qmax)]
    ee_cut = ee[(qq > qmin) & (qq < qmax)]

    # Vectorized MC noise generation (replaces Python double loop)
    noise = np.random.normal(0, ee_cut * 2.355, size=(NN, len(ee_cut)))
    ii_cut_random = ii_cut[np.newaxis, :] + noise

    LBLU = "#ccf2ff"; YEL = "#f5f794"
    plt.rc("font", size=14); plt.rc('legend', **{'fontsize': 14}); plt.rcParams['font.family'] = 'M+ 2c'
    plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; plt.rcParams['figure.figsize'] = 9.3, 6
    plt.rcParams['axes.edgecolor'] = 'r'
    plt.rcParams["figure.facecolor"] = YEL
    fig = plt.figure(facecolor=YEL)

    fig.suptitle(f"data file: {dataFile}", fontsize=14, y=1.0025)

    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1])

    ax1.set_xlabel("q$_\mathregular{z}$ (\u00c5$\mathregular{^{-1}}$)")
    ax1.set_ylabel("R/R$_\mathregular{F}$")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.yaxis.set_ticks_position('both'); ax1.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both'); ax2.xaxis.set_ticks_position('both')

    for ax in fig.get_axes():
        ax.tick_params(which='both', color='r')
        ax.set_facecolor(LBLU)

    ax1.annotate(f"q_min = {qmin}\nq_max = {qmax}", xy=(0.8, 0.8), xycoords='axes fraction', fontsize=8)

    ax2.set_ylabel("$\mathregular{\\rho}$ (e/\u00c5\u00b3)")
    ax2.set_xlabel("z (\u00c5)")

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H.%M.%S')

    # Parallelized MC fits (replaces sequential tqdm loop)
    results = Parallel(n_jobs=-1)(
        delayed(_single_mc_fit)(params, qq_cut, ii_cut_random[nn], ee_cut, fitFunc, weight, doConv, method)
        for nn in tqdm(range(NN), desc="MC fits")
    )

    # Set up post-processing arrays from first result
    result0 = results[0]
    DD = 0
    topLayer_RR = 0  # default for numbLayers == 0
    for mm in range(int(result0.params['numbLayers'].value)):
        DD += result0.params[f'layer{mm}_dd'].value
        topLayer_RR = result0.params[f'layer{mm}_sig'].value
    sub_RR = result0.params['sub_sig'].value
    zz = np.arange(0 - 7 * topLayer_RR, DD + 7 * sub_RR, 0.1)

    dens = np.zeros([NN, len(zz)])
    dens_zeroRoughness = np.zeros([NN, len(zz)])
    beta = np.zeros([NN, len(zz)])
    beta_zeroRoughness = np.zeros([NN, len(zz)])

    qq_plot = np.arange(0, np.max(qq), 0.001)
    final = np.zeros([NN, len(qq_cut)])
    ii_plot = np.zeros([NN, len(qq_plot)])

    bestFitParamsList = np.zeros([NN, len(params)])
    bestFitParam_name = [str(item[0]) for item in result0.params.items()]
    bestFitVary = [str(result0.params[item[0]].vary) for item in result0.params.items()]

    # Post-process each result sequentially
    for nn in tqdm(range(NN), desc="Post-processing"):
        result = results[nn]

        if report:
            report_fit(result)

        if fitFunc == xrr_parratt_fit:
            final[nn, :] = xrr_parratt_calc(result.params, qq_cut, doConv=doConv)
            ii_plot[nn, :] = xrr_parratt_calc(result.params, qq_plot, doConv=doConv)
        elif fitFunc == xrr_master_refractionCorrected_fit:
            final[nn, :] = xrr_master_refractionCorrected_calc(result.params, qq_cut, doConv=doConv)
            ii_plot[nn, :] = xrr_master_refractionCorrected_calc(result.params, qq_plot, doConv=doConv)

        bestFitParamsList[nn, :] = [result.params[item[0]].value for item in result.params.items()]

        dens[nn, :] = xrr_eDens(result.params, zz)
        dens_zeroRoughness[nn, :] = xrr_eDens_zeroRoughness(result.params, zz)

        if fitFunc == xrr_parratt_fit:
            beta[nn, :] = xrr_beta(result.params, zz)
            beta_zeroRoughness[nn, :] = xrr_beta_zeroRoughness(result.params, zz)

    # Plotting loop
    result = results[-1]  # use last result for rrf parameters
    if rrfPlot:
        rrf_qq      = _fresnel_rrf(qq,      result.params)
        rrf_qq_plot = _fresnel_rrf(qq_plot, result.params)
        rrf_qq_cut  = _fresnel_rrf(qq_cut,  result.params)
    for nn in tqdm(range(NN), desc="Plotting"):

        if rrfPlot:
            if nn == 0:
                ax1.semilogy(qq, ii/rrf_qq, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
                if ebar == 1:
                    ax1.errorbar(qq, ii/rrf_qq, yerr=ee/rrf_qq, linestyle='None', color='b', capsize=0, elinewidth=1)
            ax1.semilogy(qq_plot, ii_plot[nn, :]/rrf_qq_plot, color='k', linewidth=1)
            ax1.semilogy(qq_cut, final[nn, :]/rrf_qq_cut, color='r', linewidth=1)
        else:
            if nn == 0:
                ax1.semilogy(qq, ii, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
                if ebar == 1:
                    ax1.errorbar(qq, ii, yerr=ee, linestyle='None', color='b', capsize=0, elinewidth=1)
            ax1.semilogy(qq_plot, ii_plot[nn, :], color='k', linewidth=1)
            ax1.semilogy(qq_cut, final[nn, :], color='r', linewidth=1)

        if NN >= 10:
            ax2.plot(zz, dens[nn, :], color='k', linewidth=1, alpha=1/(NN/10))
        else:
            ax2.plot(zz, dens[nn, :], color='k', linewidth=1)

    # Save output files
    xrrOutput = np.vstack((qq_plot, ii_plot))
    np.savetxt(f"{outputName}_{st}_fit_mc.r", xrrOutput.T)

    edOutput = np.vstack((zz, dens))
    np.savetxt(f"{outputName}_{st}_fit_mc.ed", edOutput.T)

    nedOutput = np.vstack((zz, dens_zeroRoughness))
    np.savetxt(f"{outputName}_{st}_fit_mc.ned", nedOutput.T)

    betaOutput = np.vstack((zz, beta))
    np.savetxt(f"{outputName}_{st}_fit_mc.beta", betaOutput.T)

    nbetaOutput = np.vstack((zz, beta_zeroRoughness))
    np.savetxt(f"{outputName}_{st}_fit_mc.nbeta", nbetaOutput.T)

    paramOutput = np.vstack((bestFitParamsList, bestFitParam_name, bestFitVary))
    np.savetxt(f"{outputName}_{st}_fit_mc.fitParams", paramOutput.T, fmt="%s")

    bestFitParamsList_mean = np.mean(bestFitParamsList, axis=0)
    bestFitParamsList_std = np.std(bestFitParamsList, axis=0)

    paramOutputStats = np.vstack((bestFitParam_name,
                                  np.around(bestFitParamsList_mean, decimals=3),
                                  np.around(bestFitParamsList_std, decimals=3)))
    paramOutputStats = paramOutputStats.T

    np.savetxt(f"{outputName}_{st}_fit_mc.fitParamsStats",
               np.vstack((bestFitParamsList_mean, bestFitParamsList_std, bestFitParam_name)).T, fmt="%s")

    ax3.set_xticks([]); ax3.set_yticks([])

    table1 = PrettyTable(paramOutputStats.dtype.names)
    for row in paramOutputStats:
        table1.add_row(row)
    table1.field_names = ["parameter", "mean", "std"]
    table1.align['parameter'] = 'l'
    table1.padding_width = 7

    ax3.text(0.01, 0.01, table1, fontsize=8)

    gs.tight_layout(fig)

    plt.savefig(f"{outputName}_{st}_fit_mc.png", bbox_inches='tight', facecolor=YEL, dpi=600)


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def loadParams(parameterFilename, lowLim=0.1, highLim=0.1):
    '''
    function to load output parameter file from performFit into lmfit Parameters
    argument lowLim is the ratio of the parameter value that is subtracted to make the new minimum in the parameter for fitting
    argument highLim is the ratio of the parameter value that is added to make the new maximum in the parameter for fitting
    fixed or varied is the same as in fit the generated the file thats loaded
    '''
    ### load parameter array
    parasValue = np.loadtxt(parameterFilename, usecols=[0,])
    ### load parameter names — Fixed: np.str deprecated, use str
    parasName = np.loadtxt(parameterFilename, usecols=[1], dtype=str)
    ### load boolean of parameter varied or not during fit
    parasVary = np.loadtxt(parameterFilename, usecols=[2], dtype=str)

    ### Make lmfit Parameters
    params = Parameters()

    ### add the parameters
    for name, value, vary in zip(parasName, parasValue, parasVary):
        ### if parameters are to be varied and have limits
        if vary == 'True':
            params.add(name, value=value, min=(value - lowLim*value), max=(value + highLim*value), vary=True)
        ### if parameters are fixed
        else:
            params.add(name, value=value, vary=False)

    return params


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def analyze_mc(file, bins=20):
    '''
    function to analyze the parameter matrix generated from performFit_mc
    makes use of python corner package
    argument bin lets you choose how many bins are used in plot
    '''
    ### check how many NN in performFit_mc
    with open(file) as f:
        ncols = len(f.readline().split(' '))

    ### load parameter array
    paras = np.loadtxt(file, usecols=range(0, ncols-2))
    ### load parameter names — Fixed: np.str deprecated, use str
    parasName = np.loadtxt(file, usecols=(-2), dtype=str)
    ### load boolean of parameter varied or not during fit
    parasVary = np.loadtxt(file, usecols=(-1), dtype=str)

    ### use only varied parameters
    paras = paras[parasVary == 'True']
    parasName = parasName[parasVary == 'True']

    ### prepare figure
    plt.rc("font", size=10); plt.rcParams['font.family'] = 'M+ 2c'
    plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; plt.rcParams['figure.figsize'] = 9.3, 6
    plt.rcParams['axes.edgecolor'] = 'r'

    ### plot cornerplot
    plt.close('all')
    figure = corner.corner(paras.T, labels=parasName, bins=bins)

    ### save figure
    plt.savefig(f"{file}.analysis.png", bbox_inches='tight', dpi=600)
