#title           : foofit.py
#description     : functions for calculating and fits reflectivity data with parratt formalism or refraction corrected master formula
#                  code based on pyRefFit by Michael Klimczak and Hans-Georg Steinrueck
#author          : Hans-Georg Steinrueck
#e-mail          : hgs@slac.stanford.edu, hansgeorgsteinrueck@gmail.com
#date            : 08/18/2018
#version         : 0.2
#usage           : best to use with jupyter notebook
#notes           : I would call this a beta version, needs to be tested by colleagues
#                  The code was cross-checked with genx
#python_version  : written and tested 3.5.4.

### to-do: potentially export fit-report to output file
### to-do: work on emcee
### to-do: add best fit to mc
### to-do: add multilayer option via a lmfit Parameter generation script
## document steps in formulas
###

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pylab import *
import scipy.special
import scipy.ndimage

from scipy import optimize
from scipy.special import *
import cmath
import re

import datetime
import time

import lmfit
from lmfit import Model, Parameters
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, fit_report

from prettytable import PrettyTable
import random
from tqdm import tqdm

import timeit

import corner

from joblib import Parallel, delayed

from astropy.constants import sigma_T

# Classical electron radius in Angstroms, derived from Thomson cross-section via astropy:
#   sigma_T = (8*pi/3) * r_e^2  =>  r_e = sqrt(3*sigma_T / (8*pi))
r_e_AA = np.sqrt(3 * sigma_T.si.value / (8 * np.pi)) * 1e10
# Critical q prefactor: qc = qc_factor * sqrt(delta_rho)  [rho in e/Å³, qc in Å⁻¹]
qc_factor = 4 * np.sqrt(r_e_AA * np.pi)


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def smear_scipy_int(qq, rr, params, sig=0.002/2.35):
    '''
    function  that convolutes any array set with gaussian of width sig
    array does not need to be equally spaced
    "expands" array to ten times minimum point density, then calculates ratio before and after convolution, which is interpolated on original array
    I call this oversampling
    '''
    ### difference between datapoints
    dq = diff(qq)
    ### makes sure there is no overlapping datapoints
    dq_nonZero = dq[dq != 0]

    ### oversample array
    qq_exp = np.arange(np.min(qq), np.max(qq), np.min(dq_nonZero)/10)
    ### calculate oversample xrr
    rr_exp = xrr_parratt_calc(params, qq_exp, doConv=0)

    ### convolute
    rr_exp_smear = scipy.ndimage.gaussian_filter1d(rr_exp, sig/(qq_exp[1]-qq_exp[0]))

    ### calculate ratio
    scaling = rr_exp/rr_exp_smear
    ### interpolate onto original array
    scaling_int = np.interp(qq, qq_exp, scaling)

    ### calculate convoluted original array
    rr_smear = rr/scaling_int

    return rr_smear


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_parratt_calc(params, qq, doConv=0):

    I0 = params['I0'].value
    bkg = params['bkg'].value

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    rho = np.array([params['pre_rho'].value] +
                   [params['layer%s_rho' % nn].value for nn in range(numbLayers)] +
                   [params['sub_rho'].value])

    beta = np.array([params['pre_beta'].value] +
                    [params['layer%s_beta' % nn].value for nn in range(numbLayers)] +
                    [params['sub_beta'].value])

    sig = np.array([params['layer%s_sig' % nn].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])

    dd = np.array([0] + [params['layer%s_dd' % nn].value for nn in range(numbLayers)])

    numInterfaces = numbLayers + 1

    # --> this makes the wrong critical angle correct, gets relative change in rho and beta right
    beta -= beta[0]
    rho -= rho[0]

    # Fixed: use params['wavelength'].value (was hardcoded to 1.033)
    wavelength = params['wavelength'].value
    qs = np.sqrt(np.subtract(*np.meshgrid((qq**2), (8 * 2 * np.pi * r_e_AA * rho
                                                    - 1j * 32 * np.pi**2 * beta * 0.00000001 / wavelength**2))))

    # create empty arrays of the right shape for r, p; one row per interface, one column per q-value
    r = np.zeros((numInterfaces, qq.shape[0]), dtype=np.complex64)
    p = np.copy(r)

    # calculate reflective indexes for each interface, phase terms for each layer
    for ii in np.arange(0, numInterfaces):
        # *****NEW***** ### qs[ii]* qs[ii+1] instead of qs[ii]**2: refraction corrected roughness; doesn't make a big difference
        r[ii] = (qs[ii] - qs[ii+1]) / (qs[ii] + qs[ii+1]) * np.exp(-sig[ii]**2 * qs[ii] * qs[ii+1] / 2)
        p[ii] = np.exp(1j * qs[ii] * dd[ii])

    # recursively build the reflective index of the entire system from the bottom up
    rr = r[numInterfaces-1]

    for ii in np.arange(0, numInterfaces-1)[::-1]:
        rr = (r[ii] + rr * p[ii+1]) / (1 + r[ii] * rr * p[ii+1])

    rr = I0 * np.abs(rr)**2 + bkg

    if doConv != 0:
        rr = smear_scipy_int(qq, rr, params, sig=doConv)

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
    rho = np.array([params['layer%s_rho' % nn].value for nn in range(numbLayers)])
    dd = np.array([params['layer%s_dd' % nn].value for nn in range(numbLayers)])
    sig = np.array([params['layer%s_sig' % nn].value for nn in range(numbLayers)])

    layers = [(pre_rho, 0, 0)]
    for nn in range(numbLayers):
        layers.append((rho[nn], dd[nn], sig[nn]))
    layers.append((sub_rho, 0, sub_sig))

    rr = 0j
    depth = 0

    qc = qc_factor * np.sqrt(sub_rho - pre_rho)

    qq_p = np.sqrt(qq**2 - qc**2)
    rrf = np.abs((qq - qq_p) / (qq + qq_p))**2

    for nn in range(0, len(layers)-1):
        depth += layers[nn][1]
        rr += (layers[nn][0] - layers[nn+1][0]) * np.exp(1j*qq_p*depth) * np.exp(-qq**2 * layers[nn+1][2]**2 / 2)

    rr /= (sub_rho - pre_rho)

    rr = I0 * abs(rr)**2 * rrf + bkg

    if doConv != 0:
        rr = smear_scipy_int(qq, rr, params, sig=doConv)

    return rr


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_parratt_fit(params, qq, data, weight, ee, doConv=0):
    # Optimized: delegate to _calc instead of duplicating physics code
    rr = xrr_parratt_calc(params, qq, doConv=doConv)

    if weight == 0:  # normalized
        res = (rr - data) / data
    if weight == 1:  # linear
        res = (rr - data)
    if weight == 2:  # log
        res = np.log(rr) - np.log(data)
    if weight == 3:  # error-weighted
        res = (rr - data) / ee

    return res


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_master_refractionCorrected_fit(params, qq, data, weight, ee, doConv=0):
    # Optimized: delegate to _calc instead of duplicating physics code
    rr = xrr_master_refractionCorrected_calc(params, qq, doConv=doConv)

    if weight == 0:  # normalized
        res = (rr - data) / data
    if weight == 1:  # linear
        res = (rr - data)
    if weight == 2:  # log
        res = np.log(rr) - np.log(data)
    if weight == 3:  # error-weighted
        res = (rr - data) / ee

    return res


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_eDens(params, zz):

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    rho = np.array([params['pre_rho'].value] +
                   [params['layer%s_rho' % nn].value for nn in range(numbLayers)] +
                   [params['sub_rho'].value])

    sig = np.array([params['layer%s_sig' % nn].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])

    dd = np.array([0] + [params['layer%s_dd' % nn].value for nn in range(numbLayers)])

    ZZ = np.cumsum(dd)

    numInterfaces = numbLayers + 1

    # sum up density contributions each layer
    density = zz*0 + rho[0]
    for ii in np.arange(0, numInterfaces):
        density += (rho[ii + 1] - rho[ii]) * (1 + scipy.special.erf((zz-ZZ[ii]) / np.sqrt(2) / sig[ii])) / 2

    return density


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_eDens_zeroRoughness(params, zz):

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    rho = np.array([params['pre_rho'].value] +
                   [params['layer%s_rho' % nn].value for nn in range(numbLayers)] +
                   [params['sub_rho'].value])

    sig = np.array([params['layer%s_sig' % nn].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])
    sig = 0 * sig + 0.0001

    dd = np.array([0] + [params['layer%s_dd' % nn].value for nn in range(numbLayers)])

    ZZ = np.cumsum(dd)

    numInterfaces = numbLayers + 1

    # sum up density contributions each layer
    density = zz*0 + rho[0]
    for ii in np.arange(0, numInterfaces):
        density += (rho[ii + 1] - rho[ii]) * (1 + scipy.special.erf((zz-ZZ[ii]) / np.sqrt(2) / sig[ii])) / 2

    return density


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_beta(params, zz):

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    beta = np.array([params['pre_beta'].value] +
                    [params['layer%s_beta' % nn].value for nn in range(numbLayers)] +
                    [params['sub_beta'].value])

    sig = np.array([params['layer%s_sig' % nn].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])

    dd = np.array([0] + [params['layer%s_dd' % nn].value for nn in range(numbLayers)])

    ZZ = np.cumsum(dd)

    numInterfaces = numbLayers + 1

    # sum up absorption contributions each layer
    absorption = zz*0 + beta[0]
    for ii in np.arange(0, numInterfaces):
        absorption += (beta[ii + 1] - beta[ii]) * (1 + scipy.special.erf((zz-ZZ[ii]) / np.sqrt(2) / sig[ii])) / 2

    absorption *= 0.00000001

    return absorption


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def xrr_beta_zeroRoughness(params, zz):

    numbLayers = int(params['numbLayers'].value)

    # Optimized: list comprehensions instead of concatenate loops
    beta = np.array([params['pre_beta'].value] +
                    [params['layer%s_beta' % nn].value for nn in range(numbLayers)] +
                    [params['sub_beta'].value])

    sig = np.array([params['layer%s_sig' % nn].value for nn in range(numbLayers)] +
                   [params['sub_sig'].value])
    sig = 0 * sig + 0.0001

    dd = np.array([0] + [params['layer%s_dd' % nn].value for nn in range(numbLayers)])

    ZZ = np.cumsum(dd)

    numInterfaces = numbLayers + 1

    # sum up absorption contributions each layer
    absorption = zz*0 + beta[0]
    for ii in np.arange(0, numInterfaces):
        absorption += (beta[ii + 1] - beta[ii]) * (1 + scipy.special.erf((zz-ZZ[ii]) / np.sqrt(2) / sig[ii])) / 2

    absorption *= 0.00000001
    return absorption


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
def performFit(dataFile, params, fitFunc=xrr_parratt_fit, method='differential_evolution',
               qmin=0, qmax=1, plot=1, report=True, save=True, outputName="foo", weight=0, rrfPlot=False,
               doConv=0):

    close('all')

    data = np.loadtxt(dataFile)

    qq = data[:,0]
    ii = data[:,1]
    try:
        ee = data[:,2]
        ebar = 1
    except:
        ee = np.zeros(len(ii))
        ebar = 0

    qq_cut = qq[(qq > qmin) & (qq < qmax)]
    ii_cut = ii[(qq > qmin) & (qq < qmax)]
    ee_cut = ee[(qq > qmin) & (qq < qmax)]

    start = timeit.default_timer()

    minner = Minimizer(fitFunc, params, fcn_args=(qq_cut, ii_cut, weight, ee_cut, doConv), nan_policy='omit')
    result = minner.minimize(method=method)

    stop = timeit.default_timer()
    print("time for fit: %0.2f s" % (stop - start))

    # calculate final result
    if fitFunc == xrr_parratt_fit:
        final = xrr_parratt_calc(result.params, qq_cut, doConv=doConv)
    if fitFunc == xrr_master_refractionCorrected_fit:
        final = xrr_master_refractionCorrected_calc(result.params, qq_cut, doConv=doConv)

    qq_plot = np.arange(0, np.max(qq), 0.001)
    if fitFunc == xrr_parratt_fit:
        ii_plot = xrr_parratt_calc(result.params, qq_plot, doConv=doConv)
    if fitFunc == xrr_master_refractionCorrected_fit:
        ii_plot = xrr_master_refractionCorrected_calc(result.params, qq_plot, doConv=doConv)

    # write error report
    if report:
        report_fit(result)

    if plot == 1:

        LBLU = "#ccf2ff"; YEL = "#f5f794"
        plt.rc("font", size=14); plt.rc('legend', **{'fontsize': 14}); plt.rcParams['font.family'] = 'M+ 2c'
        plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; rcParams['figure.figsize'] = 9.3, 6
        plt.rcParams['axes.edgecolor'] = 'r'
        plt.rcParams["figure.facecolor"] = YEL
        fig = figure(facecolor=YEL)

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
            rrf = abs((qq - sqrt(qq**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq + np.sqrt(qq**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq, ii/rrf, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
            if ebar == 1:
                ax1.errorbar(qq, ii/rrf, yerr=ee/rrf, linestyle='None', color='b', capsize=0, elinewidth=1)
            rrf = abs((qq_plot - np.sqrt(qq_plot**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq_plot + np.sqrt(qq_plot**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq_plot, ii_plot/rrf, color='k', linewidth=1)
            rrf = abs((qq_cut - np.sqrt(qq_cut**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq_cut + np.sqrt(qq_cut**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq_cut, final/rrf, color='r', linewidth=1)
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
        np.savetxt("%s_%s_fit.r" % (outputName, st), xrrOutput)

        paramOutput = np.column_stack((bestFitParam, bestFitParam_name, bestFitVary))
        np.savetxt("%s_%s_fit.fitParams" % (outputName, st), paramOutput, fmt="%s")

        LBLU = "#ccf2ff"; YEL = "#f5f794"
        plt.rc("font", size=14); plt.rc('legend', **{'fontsize': 14}); plt.rcParams['font.family'] = 'M+ 2c'
        plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; rcParams['figure.figsize'] = 9.3, 6
        plt.rcParams['axes.edgecolor'] = 'r'
        plt.rcParams["figure.facecolor"] = YEL
        fig = figure(facecolor=YEL)

        fig.suptitle("data file: %s" % dataFile, fontsize=14, y=1.0025)

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
            rrf = abs((qq - sqrt(qq**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq + np.sqrt(qq**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq, ii/rrf, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
            if ebar == 1:
                ax1.errorbar(qq, ii/rrf, yerr=ee/rrf, linestyle='None', color='b', capsize=0, elinewidth=1)
            rrf = abs((qq_plot - np.sqrt(qq_plot**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq_plot + np.sqrt(qq_plot**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq_plot, ii_plot/rrf, color='k', linewidth=1)
            rrf = abs((qq_cut - np.sqrt(qq_cut**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq_cut + np.sqrt(qq_cut**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq_cut, final/rrf, color='r', linewidth=1)
        else:
            ax1.semilogy(qq, ii, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
            if ebar == 1:
                ax1.errorbar(qq, ii, yerr=ee, linestyle='None', color='b', capsize=0, elinewidth=1)
            ax1.semilogy(qq_plot, ii_plot, color='k', linewidth=1)
            ax1.semilogy(qq_cut, final, color='r', linewidth=1)

        ax1.annotate("q_min = %s\nq_max = %s" % (qmin, qmax), xy=(0.8, 0.8), xycoords='axes fraction', fontsize=8)

        ax2.set_ylabel("$\mathregular{\\rho}}$ (e/\u00c5\u00b3)")
        ax2.set_xlabel("z (\u00c5)")

        DD = 0
        for nn in range(int(result.params['numbLayers'].value)):
            DD += result.params['layer%s_dd' % nn].value
            topLayer_RR = result.params['layer%s_sig' % nn].value
        sub_RR = result.params['sub_sig'].value
        zz = np.arange(0-7*topLayer_RR, DD+7*sub_RR, 0.01)

        dens = xrr_eDens(result.params, zz)
        dens_zeroRoughness = xrr_eDens_zeroRoughness(result.params, zz)

        edOutput = np.column_stack((zz, dens))
        np.savetxt("%s_%s_fit.ed" % (outputName, st), edOutput)

        nedOutput = np.column_stack((zz, dens_zeroRoughness))
        np.savetxt("%s_%s_fit.ned" % (outputName, st), nedOutput)

        if fitFunc == xrr_parratt_fit:
            beta_z = xrr_beta(result.params, zz)
            beta_zeroRoughness = xrr_beta_zeroRoughness(result.params, zz)

            betaOutput = np.column_stack((zz, beta_z))
            np.savetxt("%s_%s_fit.beta" % (outputName, st), betaOutput)

            nbetaOutput = np.column_stack((zz, beta_zeroRoughness))
            np.savetxt("%s_%s_fit.nbeta" % (outputName, st), nbetaOutput)

        ax2.plot(zz, dens, color='k', linewidth=1)
        ax2.plot(zz, dens_zeroRoughness, color='k', linewidth=1, linestyle='dashed')

        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.text(0.01, 0.01, fit_report(result), fontsize=8)

        gs.tight_layout(fig)

        if save:
            savefig("%s_%s_fit.png" % (outputName, st), bbox_inches='tight', facecolor=YEL, dpi=600)


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

    close('all')

    data = np.loadtxt(dataFile)

    qq = data[:,0]
    ii = data[:,1]
    try:
        ee = data[:,2]
        ebar = 1
    except:
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
    plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; rcParams['figure.figsize'] = 9.3, 6
    plt.rcParams['axes.edgecolor'] = 'r'
    plt.rcParams["figure.facecolor"] = YEL
    fig = figure(facecolor=YEL)

    fig.suptitle("data file: %s" % dataFile, fontsize=14, y=1.0025)

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

    ax1.annotate("q_min = %s\nq_max = %s" % (qmin, qmax), xy=(0.8, 0.8), xycoords='axes fraction', fontsize=8)

    ax2.set_ylabel("$\mathregular{\\rho}}$ (e/\u00c5\u00b3)")
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
        DD += result0.params['layer%s_dd' % mm].value
        topLayer_RR = result0.params['layer%s_sig' % mm].value
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
        if fitFunc == xrr_master_refractionCorrected_fit:
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
    for nn in tqdm(range(NN), desc="Plotting"):

        if rrfPlot:
            if nn == 0:
                rrf = abs((qq - sqrt(qq**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                          (qq + np.sqrt(qq**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
                ax1.semilogy(qq, ii/rrf, linestyle='none', marker='o', color='b', zorder=-32, markersize=3)
                if ebar == 1:
                    ax1.errorbar(qq, ii/rrf, yerr=ee/rrf, linestyle='None', color='b', capsize=0, elinewidth=1)
            rrf = abs((qq_plot - np.sqrt(qq_plot**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq_plot + np.sqrt(qq_plot**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq_plot, ii_plot[nn, :]/rrf, color='k', linewidth=1)
            rrf = abs((qq_cut - np.sqrt(qq_cut**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)) /
                      (qq_cut + np.sqrt(qq_cut**2 - (qc_factor*np.sqrt(result.params['sub_rho'].value - result.params['pre_rho'].value))**2)))**2
            ax1.semilogy(qq_cut, final[nn, :]/rrf, color='r', linewidth=1)
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
    np.savetxt("%s_%s_fit_mc.r" % (outputName, st), xrrOutput.T)

    edOutput = np.vstack((zz, dens))
    np.savetxt("%s_%s_fit_mc.ed" % (outputName, st), edOutput.T)

    nedOutput = np.vstack((zz, dens_zeroRoughness))
    np.savetxt("%s_%s_fit_mc.ned" % (outputName, st), nedOutput.T)

    betaOutput = np.vstack((zz, beta))
    np.savetxt("%s_%s_fit_mc.beta" % (outputName, st), betaOutput.T)

    nbetaOutput = np.vstack((zz, beta_zeroRoughness))
    np.savetxt("%s_%s_fit_mc.nbeta" % (outputName, st), nbetaOutput.T)

    paramOutput = np.vstack((bestFitParamsList, bestFitParam_name, bestFitVary))
    np.savetxt("%s_%s_fit_mc.fitParams" % (outputName, st), paramOutput.T, fmt="%s")

    bestFitParamsList_mean = np.mean(bestFitParamsList, axis=0)
    bestFitParamsList_std = np.std(bestFitParamsList, axis=0)

    paramOutputStats = np.vstack((bestFitParam_name,
                                  np.around(bestFitParamsList_mean, decimals=3),
                                  np.around(bestFitParamsList_std, decimals=3)))
    paramOutputStats = paramOutputStats.T

    np.savetxt("%s_%s_fit_mc.fitParamsStats" % (outputName, st),
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

    savefig("%s_%s_fit_mc.png" % (outputName, st), bbox_inches='tight', facecolor=YEL, dpi=600)


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
    plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in'; rcParams['figure.figsize'] = 9.3, 6
    plt.rcParams['axes.edgecolor'] = 'r'

    ### plot cornerplot
    close('all')
    figure = corner.corner(paras.T, labels=parasName, bins=bins)

    ### save figure
    savefig("%s.analysis.png" % file, bbox_inches='tight', dpi=600)
