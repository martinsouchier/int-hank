"""
This file contains functions used to calibrate, solve the model and display results
"""

import numpy as np
from numba import njit
import scipy.optimize as opt
import pandas as pd
import aux_model as mod
import matplotlib.pyplot as plt

""" Calibration """

def mpcs(c, a, a_grid, coh):
    
    """Approximate mpc, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    
    # initialize
    mpcs_ = np.empty_like(c)

    # symmetric differences away from boundaries
    mpcs_[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (coh[:, 2:] - coh[:, :-2])

    # asymmetric first differences at boundaries
    mpcs_[:, 0] = (c[:, 1] - c[:, 0]) / (coh[:, 1] - coh[:, 0])
    mpcs_[:, -1] = (c[:, -1] - c[:, -2]) / (coh[:, -1] - coh[:, -2])

    # special case of constrained
    mpcs_[a == a_grid[0]] = 1

    return mpcs_

def decile_make(ss, mpc_e, D, cHi, cFi, incidence_i, inc_i, nbin = 10):

    """Compute statistics per income decile"""

    i=0
    mass = 0
    centile = 1/nbin
    mpc_decile_model = np.zeros(nbin)
    cH_decile_model = np.zeros(nbin)
    cF_decile_model = np.zeros(nbin)
    incidence_decile_model = np.zeros(nbin)
    inc_decile = np.zeros(nbin)
    income_threshold = np.zeros(nbin)
    mpc = 0
    cH = 0
    cF = 0
    incidence = 0
    inc = 0
    De = np.sum(D, axis=1)
    Dtemp = D.copy()
    for j in range(nbin):

        cont = True
        while cont == True:

            # if do not reach 10% yet, add the mass and move on
            if (mass + De[i] < centile) & (i < ss['nE']-1):
                mpc += np.sum(mpc_e[i]*Dtemp[i,:])/centile                          # need to divide by mass per bin
                cH += np.sum(cHi[i,:]*Dtemp[i,:])/centile
                cF += np.sum(cFi[i,:]*Dtemp[i,:])/centile
                incidence += np.sum(incidence_i[i,:]*Dtemp[i,:])/centile
                inc += np.sum(inc_i[i,:]*Dtemp[i,:])/centile
                mass += De[i]
                De[i] = 0                 # empty distribution so that we do not double count for the next bin
                i += 1
                frac = 0
            # otherwise, only add a fraction of the mass
            else:
                frac = (centile-mass)/De[i]                              # keep a fraction of the mass s.t. bin reaches 10% exactly
                mpc += frac*np.sum(mpc_e[i]*Dtemp[i,:])/centile
                cH += frac*np.sum(cHi[i,:]*Dtemp[i,:])/centile
                cF += frac*np.sum(cFi[i,:]*Dtemp[i,:])/centile
                incidence += frac*np.sum(incidence_i[i,:]*Dtemp[i,:])/centile
                inc += frac*np.sum(inc_i[i,:]*Dtemp[i,:])/centile
                cont = False                                              # break the while loop
                De[i] = (1-frac)*De[i]                                       # remove the fraction of the distribution already allocated
                Dtemp[i,:] = (1-frac)*Dtemp[i,:]                             
                mpc_decile_model[j] = mpc                                  # store values for that bin
                cH_decile_model[j] = cH
                cF_decile_model[j] = cF
                incidence_decile_model[j] = incidence
                inc_decile[j] = inc
                income_threshold[j] = i
                mass = 0                                                  # re-initialize
                mpc = 0
                tradeshare = 0
                cH = 0
                cF = 0
                incidence = 0
                inc = 0
    
    # Compute share of tradable per income decile
    tradeshare_decile_model = cF_decile_model/(cF_decile_model+cH_decile_model)
                
    # Compute share of each group in overall consumption
    cHshare_decile_model = cH_decile_model/np.sum(cH_decile_model)
    
    # Compute share of income in overal income
    incshare_decile = inc_decile/np.sum(inc_decile)
    
    return mpc_decile_model, tradeshare_decile_model, cHshare_decile_model, incidence_decile_model, incshare_decile, income_threshold

def calibration_mpc(ss, T = 200, show_incdec=True):

    """Calibrate model using data per income decile"""

    # get the micro data
    microdata = pd.read_excel('data/data_final.xlsx',sheet_name='Data',nrows=10)
    dta = microdata.values
    decile = dta[:,0]
    cHshare_decile = dta[:,2]
    tradeshare_decile = dta[:,3]
    incidence_decile = dta[:,4]
    mpc_decile = dta[:,5]
    incshare_decile_data = dta[:,6]
    
    # Adjust incidence in data so that weighted average is 1
    cste = np.sum(incidence_decile*incshare_decile_data)
    incidence_decile = incidence_decile/cste
    
    # average mpc and average tradable share
    mpc_avg_data = np.mean(mpc_decile)
    tradeshare_avg_data = 0.4           # by construction; np.mean(tradeshare_decile)

    # Compute MPCs out of labor income and out of capital income
    ss['M'] = np.ones((ss['n_exog'],ss['nA']))
    JCw = mod.hh_HA.jacobian(ss, inputs=['n'], T=T)['C']['n']*ss['markup_ss']
    JCTf = mod.hh_HA.solve_jacobian(ss, [], [], ['Transfer'], T=T)['C']['Transfer']
    JCr0 = mod.hh_HA.solve_jacobian(ss, [], [], ['rpost'], T=T)['C']['rpost']
    
    # Statistics per income decile

    # Compute mpc per income bin
    mpc_e = np.zeros(ss['n_exog'])
    for i in range(ss['n_exog']):

        # Update incidence matrix M
        ss['M'] = np.zeros((ss['n_exog'],ss['nA']));
        ss['M'][i,:] = 1

        # Compute annual MPC per income group
        mpc_e[i] = np.sum(mod.hh_HA.solve_jacobian(ss, [], [], ['Transfer'], T=T)['C']['Transfer'][:1,0])/(ss['pi_e'][i]/ss['n_beta'])

    # Compute moments per income decile and beta
    mpc_decile_beta, tradeshare_decile_beta, cHshare_decile_beta, incidence_decile_beta, incshare_decile_beta = np.zeros((5,ss['n_beta'],10))
    for i in range(ss['n_beta']):
        idx = np.arange(ss['nE'])*ss['n_beta'] + i
        mpc_b = mpc_e[idx]
        D = ss['n_beta']*ss.internals['hh_HA']['D'][idx]
        cHi = ss['cHi'][idx]
        cFi = ss['cFi'][idx]
        incidence_i = ss['incidence_i'][idx]
        inc_i = ss['e_grid'][idx]
        inc_i = np.tile(inc_i,(ss['nA'],1)).T
        mpc_decile_beta[i], tradeshare_decile_beta[i], cHshare_decile_beta[i], incidence_decile_beta[i], incshare_decile_beta[i], _ = decile_make(ss, mpc_b, D, cHi, cFi, incidence_i, inc_i)

    # Compute moments per income decile
    mpc_decile_model = np.mean(mpc_decile_beta,0)
    tradeshare_decile_model = np.mean(tradeshare_decile_beta,0)
    cHshare_decile_model = np.mean(cHshare_decile_beta,0)
    incidence_decile_model = np.mean(incidence_decile_beta,0)
    incshare_decile_model = np.mean(incshare_decile_beta,0)

    # Cross-sectional deviation of income
    sd_e = np.sqrt((ss['pi_e']/ss['n_beta'])@(ss['e_grid']**2)-((ss['pi_e']/ss['n_beta'])@ss['e_grid'])**2)

    # mpc
    mpc_avg_model = np.mean(mpc_decile_model)
    mpc_avg_data = np.mean(mpc_decile)
    mpc_sd_model = np.sqrt(np.mean(mpc_decile_model**2) - np.mean(mpc_decile_model)**2)
    mpc_sd_data = np.sqrt(np.mean(mpc_decile**2) - np.mean(mpc_decile)**2)

    # budget shares
    tradeshare_avg_model = np.mean(ss['cF']/(ss['cF']+ss['cH']))
    if ss['cbarF'] > 0: 
        tradeshare_sd_model = np.sqrt(np.mean(tradeshare_decile_model**2) - np.mean(tradeshare_decile_model)**2)
    else:
        tradeshare_sd_model = 0
    tradeshare_sd_data = np.sqrt(np.mean(tradeshare_decile**2) - np.mean(tradeshare_decile)**2)

    # incidence
    incidence_avg_model = np.mean(incidence_decile_model)
    incidence_avg_data = np.mean(incidence_decile)
    incidence_sd_model = np.sqrt(np.mean(incidence_decile_model**2) - np.mean(incidence_decile_model)**2)
    incidence_sd_data = np.sqrt(np.mean(incidence_decile**2) - np.mean(incidence_decile)**2)

    if ss['cbarF'] > 0: 

        # Correlation between mpc and budget shares
        cov_mpc_tradeshare_model = np.mean(mpc_decile_model*tradeshare_decile_model) - mpc_avg_model*tradeshare_avg_model
        cov_mpc_tradeshare_data = np.mean(mpc_decile*tradeshare_decile) - mpc_avg_data*tradeshare_avg_data
        corr_mpc_tradeshare_model = cov_mpc_tradeshare_model/(tradeshare_sd_model*mpc_sd_model)
        corr_mpc_tradeshare_data = cov_mpc_tradeshare_data/(tradeshare_sd_data*mpc_sd_data)

    else: 

        # Correlation between mpc and budget shares
        cov_mpc_tradeshare_data = np.mean(mpc_decile*tradeshare_decile) - mpc_avg_data*tradeshare_avg_data
        corr_mpc_tradeshare_data = cov_mpc_tradeshare_data/(tradeshare_sd_data*mpc_sd_data)
        corr_mpc_tradeshare_model = 0
        
    # keep the data for the plots
    dta_plots = {}
    dta_plots['mpc_decile_model'] = mpc_decile_model
    dta_plots['mpc_decile'] = mpc_decile
    dta_plots['tradeshare_decile_model'] = tradeshare_decile_model
    dta_plots['tradeshare_decile'] = tradeshare_decile
    dta_plots['cHshare_decile_model'] = cHshare_decile_model
    dta_plots['cHshare_decile'] = cHshare_decile
    dta_plots['incidence_decile_model'] = incidence_decile_model
    dta_plots['incidence_decile'] = incidence_decile
    dta_plots['incshare_decile_model'] = incshare_decile_model
    dta_plots['incshare_decile'] = incshare_decile_data
    
    if show_incdec == True:
            
        print('Model: quarterly mpc out of labor income = % .4f; out of transfers = % .4f; out of revaluation effects = % .4f' %(JCw[0,0], JCTf[0,0], JCr0[0,0]/ss['A']))
        #print('\nCross-sectional standard deviation of productivity = % .4f' %(sd_e))
        print(f'Average mpc: model = {mpc_avg_model:.4}; data = {mpc_avg_data:.4}')
        #print(f'Standard deviation of mpc: model = {mpc_sd_model:.4}; data = {mpc_sd_data:.4}')
        print(f'Average import share: model = {tradeshare_avg_model:.4}; data = {tradeshare_avg_data:.4}')
        if ss['cbarF'] > 0: 
            print(f'Standard deviation of import share: model = {tradeshare_sd_model:.4}; data = {tradeshare_sd_data:.4}')
        if ss['zeta_e'] != 0:
            print(f'Standard deviation of incidence: model = {incidence_sd_model:.4}; data = {incidence_sd_data:.4}')

    return sd_e, mpc_avg_model, mpc_avg_data, mpc_sd_model, mpc_sd_data, tradeshare_avg_model, tradeshare_avg_data, tradeshare_sd_model, tradeshare_sd_data, corr_mpc_tradeshare_model, corr_mpc_tradeshare_data, JCw[0,0], JCTf[0,0], JCr0[0,0]/ss['A'], dta_plots, mpc_decile_beta, incidence_sd_model, incidence_sd_data

@njit
def ar1_simul(ρ,σ,T=10000000,T0=10000):

    """Simulate income process"""

    # Load shocks
    np.random.seed(12345)
    eps = np.random.normal(0,1,T0+T)

    # Simulate series forward
    x = np.zeros((T0+T))
    for t in range(1,T0+T):
        x[t] = ρ*x[t-1] + σ*eps[t]

    # Get ride of initial T0 periods
    x = x[T0:]

    # Compute annual averages
    x_annual = x.copy()
    x_annual = np.reshape(x_annual,(int(T/4),4))
    x_annual = np.exp(x_annual)                        # transform in level of income
    x_annual = np.sum(x_annual,1)/4
    x_annual = np.log(x_annual)                        # take logs on annual averages

    return np.corrcoef(x_annual[1:],x_annual[:-1])[0,1], np.std(x_annual), np.std(x)

""" Impulse responses """

def rshock(sd,rho,ss,T,shock_type,Q0=False):

    """Compute path of shocks"""

    # Compute path for dQ
    dr = rho**np.arange(T)
    dQ = (1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr)[0]

    # Adjust path of r to hit specific value of dQ at t=0
    if Q0 == False: 

        if shock_type == 'rstar':
            dr = sd*dr
            dQ = 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr
            
        elif shock_type == 'r':
            dr = sd*dr
            dQ = - 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr

    else:

        if shock_type == 'rstar':
            dr = sd*dr*Q0/(1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr)[0]
            dQ = 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr
            
        elif shock_type == 'r':
            dr = sd*dr*Q0/(1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr)[0]
            dQ = - 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr

    return dr, dQ

def irf_make(G,shock,shockname,varlist,T):

    """Compute IRFs"""

    irf= {}
    for i, var in enumerate(varlist): 
        if var == shockname:
            irf[var] = shock
        else: 
            try:
                irf[var] = G[var][shockname] @ shock
            except:
                irf[var] = np.zeros(T)

    return irf

def norm_fig(ss,var,nonorm = []):

    """Normalize by steady state values"""

    norm = {k: ss[k] for k in var}
    norm.update({k: 1 for k in nonorm})
    return norm
