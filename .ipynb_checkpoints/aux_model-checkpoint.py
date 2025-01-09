"""
This file contains model blocks and functions
"""

import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.utilities import discretize, interpolate, optimized_routines
from sequence_jacobian.blocks.auxiliary_blocks import jacobiandict_block
import scipy.optimize as opt
from sequence_jacobian.blocks.support.het_support import CombinedTransition, ForwardShockableTransition

""" HA block """

def hh_init(rpost, w, n, cbarF, eis, a_grid, e_grid, M, Transfer):
    """ initialize guess for policy function iteration """
    Tf = - cbarF
    coh = (1 + rpost) * a_grid + w * n * e_grid[:, np.newaxis] + Tf + M*Transfer
    Va = (1 + rpost) * (0.2 * coh) ** (-1 / eis)
    return Va, coh

@sj.het(exogenous='Pi', policy='a', backward='Va',backward_init=hh_init)
def hh_HA(Va_p, a_grid, e_grid, coh, rsub_ha, beta, eis):

    """
    Single backward iteration step using endogenous gridpoint method for households with CRRA utility.
    """

    # Solve HH problem
    uc_nextgrid = (beta) * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + rsub_ha) * c ** (-1 / eis)
    
    # Compute uce
    uce = c ** (-1 / eis) * e_grid[:,np.newaxis]
    
    return Va, a, c, uce

def hetinput(e_grid, w, n, rpost, rsub, M, Transfer, pfh, cbarF, markup_ss, a_grid, zeta_e, pi_e):
    """ Define inputs that go into household's problem """
    atw_rel = markup_ss*w*n
    Tf = - pfh*cbarF
    incrisk1 = (e_grid[:, np.newaxis])**(zeta_e*np.log(atw_rel))/ np.vdot(e_grid ** (1+zeta_e*np.log(atw_rel)), pi_e)
    labinc = e_grid[:, np.newaxis]*incrisk1*w*n
    coh = (1 + rpost) * a_grid + labinc + Tf + M*Transfer
    rsub_ha = rsub
    return coh, rsub_ha

# attach input to HA block
hh_HA = hh_HA.add_hetinputs([hetinput])

""" Jacobians for baseline model """

def get_M(ss,T):
    JCw = hh_HA.jacobian(ss, inputs=['n'], T=T)['C']['n']*ss['markup_ss']
    JCr0 = hh_HA.jacobian(ss, inputs=['rpost'], T=T)['C']['rpost'][:,0]
    Jd = 1/ss['j']*(1/(1+ss['r']))**np.arange(T)
    Md = JCr0[:,np.newaxis]*Jd[np.newaxis,:]
    M = ((1-1/ss['markup_ss'])*Md + 1/ss['markup_ss'] * JCw)
    return M, Md, JCw

def get_Mr(ss,T,hblock=hh_HA):
    JCrsub = hblock.jacobian(ss, inputs=['rsub'], T=T)['C']['rsub']
    Jrsubr = rsimple.jacobian(ss, inputs=['r'], T=T)['rsub']['r']
    JCrpost = hblock.jacobian(ss, inputs=['rpost'], T=T)['C']['rpost']
    JCr0 = hblock.jacobian(ss, inputs=['rpost'], T=T)['C']['rpost'][:,0]
    JCr = JCrsub@Jrsubr + JCrpost@np.diag(np.ones(T-1),1).T         # first term is sub. effect; second term is income effect
    Jr = - 1*(1/(1+ss['r']))**np.arange(1,T+1)           
    Mr = JCr + (JCr0[:,np.newaxis]*Jr[np.newaxis,:])       # first term is the sub. + income effects; second term is the revaluation effect
    return Mr

""" Baseline model """

# sectoral demand
@sj.simple
def hh_outputs(C, pfh, phh, eta, alpha, cbarF):
    cH = (1-alpha) * phh**(-eta) * C
    cF = cbarF + alpha * pfh**(-eta) * C
    CT = cH + cF
    return cH, cF, CT

# foreign demand
@sj.simple
def foreign_c(phf,  alphastar, gamma, Cstar, eps_dcp):
    cHstar = alphastar*phf**(-gamma*eps_dcp) * Cstar
    return cHstar

@sj.solved(unknowns={'Q': (0.1,2)}, targets=['uip'], solver="brentq")
def UIP(ruip, Q, rstar, eta, alpha):
    uip = Q / Q(+1) * (1+ruip) - (1+rstar)
    rstar_out = rstar
    if eta == 1:
        phf = Q**(-1/(1-alpha))
        phh = Q**(-alpha/(1-alpha))
        pfh = Q
    else: 
        phf = ((Q**(eta-1) - alpha)/(1-alpha))**(1/(1-eta))
        phh = ((1-alpha*Q**(1-eta))/(1-alpha))**(1/(1-eta))
        pfh = Q
    return uip, phf, phh, pfh, rstar_out

@sj.solved(unknowns={'J': 15., 'j': 15.}, targets=['Jres','jres'], solver="broyden_custom")
def income(y,Z,phh,J,j,rinc,dividend_X,pcX_home,markup_ss):
    dividend = (1-1/markup_ss)*y*phh
    div_tot = dividend
    if pcX_home == 1: div_tot += dividend_X
    Jres = div_tot + J(1) / (1 + rinc) - J
    jres = J(1) / (1 + rinc) - j
    n = y/Z
    w = Z*phh/markup_ss
    atw_n = w*n
    atw = atw_n/n
    return jres, Jres, atw_n, dividend, atw, w, n

@sj.simple
def revaluation(J,j,rpost_shock):
    rpost = J/j(-1) - 1 + rpost_shock
    return rpost
    
@sj.simple
def rsimple(r):
    rsub = r(-1)
    rinc = r
    ruip = r
    return rsub, rinc, ruip

@sj.simple
def profitcenters(Q,phh,cHstar,eps_dcp):
    dividend_X = (Q**(1-eps_dcp)*phh**(eps_dcp) - phh)*cHstar
    return dividend_X

@sj.solved(unknowns={'nfa': (-2,2)}, targets=['nfares'], solver="brentq")
def CA(nfa, pfh, phh, y, cF, cH, rpost, dividend_X, pcX_home):
    div_tot = 0
    if pcX_home == 1: div_tot += dividend_X
    nfares = phh*y - pfh*cF - phh*cH + div_tot + rpost*nfa(-1) + nfa(-1) - nfa
    netexports = phh*y - pfh*cF - phh*cH + div_tot
    return nfares, netexports

@sj.solved(unknowns={'piw': (-2,2)}, targets=['piwres'], solver="brentq")
def unions(n, UCE, CT, atw, piw, kappa_w, markup_ss, beta, frisch, eis, vphi):
    #Cstar = UCE ** (-eis)
    Cstar = CT
    piwres = piw - beta*piw(1) - kappa_w*markup_ss*(vphi*n**(1/frisch) * Cstar**(1/eis) / atw - 1/markup_ss)
    return piwres

@sj.simple
def goods_market(y, cH, cHstar):
    goods_clearing = cH + cHstar - y
    return goods_clearing

@sj.simple
def assets_market(A, nfa, j, B):
    assets_clearing = A - nfa - j - B
    return assets_clearing

""" Delayed substitution """

@sj.solved(unknowns={'xstar': 1,'x': 1}, targets=['xstarres','xres'], solver="brentq")
def xrule(phh,xstar,x,eta,alpha,theta_share,beta):
    xstarres = xstar-xstar.ss + (1-alpha)*eta*(1-beta*theta_share)*(phh-phh.ss) - beta*theta_share*(xstar(1)-xstar.ss)
    xres = x - (1-theta_share)*xstar - theta_share*x(-1)
    return xstarres, xres

@sj.solved(unknowns={'xstar_F': 1,'x_F': 1}, targets=['xstarres_F','xres_F'], solver="brentq")
def xrule_foreign(phf,xstar_F,x_F,gamma,alphastar,theta_share,beta_star):
    xstarres_F = xstar_F-xstar_F.ss + alphastar*gamma*(1-beta_star*theta_share)*(phf-phf.ss) - beta_star*theta_share*(xstar_F(1)-xstar_F.ss)
    xres_F = x_F - (1-theta_share)*xstar_F - theta_share*x_F(-1)
    return xstarres_F, xres_F

def x_to_xF(x,eta,alpha):
    if eta != 1:
        xF = (1-(1-alpha)**(1/eta)*x**(1-1/eta))**(eta/(eta-1))*alpha**(1/(1-eta))
    else:
        xF = alpha*(1-alpha)**((1-alpha)/alpha)*x**(-(1-alpha)/alpha)
    return xF

@sj.simple
def hh_outputs_ds(C, x, alpha, eta, cbarF):
    cH = x*C
    cF = cbarF + x_to_xF(x,eta,alpha)*C
    CT = cH + cF
    return cH, cF, CT

@sj.simple
def foreign_c_ds(x_F, Cstar):
    cHstar = x_F * Cstar
    return cHstar

""" Quantitative model """

@sj.solved(unknowns={'Q': (0.1,2)}, targets=['uip'], solver="brentq")
def UIP_quant(ruip, Q, P, rstar):
    uip = Q / Q(+1) * (1+ruip) - (1+rstar)
    E = Q*P
    return uip, E

@sj.solved(unknowns={'J': 15.,'j': 15.}, targets=['Jres','jres'], solver="broyden_custom")
def income_quant(y,w,Z,phh,J,j,rinc,dividend_X,pcX_home):
    n = y/Z
    dividend = y*phh - w*n
    div_tot = dividend
    if pcX_home == 1: div_tot += dividend_X
    atw_n = w * n
    atw = w
    Jres = div_tot + J(1) / (1 + rinc) - J
    jres = J(1) / (1 + rinc) - j
    return jres, Jres, atw, n, atw_n, dividend, div_tot

@sj.simple
def revaluation_quant(q,qH,J,j,i,E,pi,rstar,rsub,delta,rpost_shock,f_firm,f_F,foreign_owned):
    rpost_firm = J/j(-1) - 1 + rpost_shock
    rpost_F = (1 + delta * q)*E/(q(-1)*E(-1)*(1+pi)) - 1     # foreign long bonds
    rpost_H = (1 + delta * qH)/(qH(-1)*(1+pi)) - 1           # local long bonds
    rpost1 = f_firm*rpost_firm + f_F*rpost_F + (1-f_firm-f_F)*rpost_H
    rpost = (1-foreign_owned)*rpost1 + foreign_owned*rsub
    return rpost, rpost_F, rpost_H, rpost_firm

@sj.solved(unknowns={'i': 0}, targets=['ires'], solver="brentq")
def taylor(i,pi,piHH,phi_pi,phi_piHH,phi_pinext,phi_i,rss,ishock,realrate):
    if realrate == 0: 
        istar = rss + phi_piHH*piHH + phi_pi*pi
        ires = (1-phi_i)*istar + phi_i*i(-1) - i + ishock
    else:
        istar = rss + phi_pinext*pi(1)
        ires = (1-phi_i)*istar + phi_i*i(-1) - i + ishock
    return ires

@sj.solved(unknowns={'B':0}, targets=['Bres'], solver="brentq")
def fiscal(B, rinc, rpost_F, rpost_H, Bbar, rho_B):
    Bres = rho_B*(B(-1)-(rpost_F - rpost_H)*Bbar) - B
    Transfer = B - (1+rinc(-1))*B(-1) + (rpost_F - rpost_H)*Bbar
    return Transfer, Bres

@sj.solved(unknowns={'q': (1,25),'qH': (1,25)}, targets=['qres','qHres'], solver="brentq")
def longbonds(q, qH, rstar, i, delta):
    qres = q - (1 + delta * q(+1))/(1 + rstar)
    qHres = qH - (1 + delta * qH(+1))/(1 + i)
    return qres, qHres

@sj.solved(unknowns={'nfa': (-2,2)}, targets=['nfares'], solver="brentq")
def CA_quant(nfa, pfh, phh, y, cF, cH, rpost, rinc, dividend_X, pcX_home, rpost_F, rpost_H, a_F, a_H, Bbar):
    div_tot = 0
    if pcX_home == 1: div_tot += dividend_X
    nfares = phh*y - pfh*cF - phh*cH + div_tot + rpost(-1)*nfa(-1) + nfa(-1) - nfa + a_F*(rpost_F-rinc(-1)) + a_H*(rpost_H-rinc(-1)) + Bbar*(rpost_F - rpost_H)
    netexports = phh*y - phh*cH - pfh*cF
    return nfares, netexports

@sj.simple
def profitcenters_quant(Q,phh,phf,cHstar):
    dividend_X = (Q*phf-phh)*cHstar
    return dividend_X

@sj.solved(unknowns={'P':1}, targets=['Pres'], solver="brentq")
def pi_to_P(P, pi):
    Pres = P - P(-1) - pi
    return Pres

@sj.solved(unknowns={'piFH': 0,'PFH': 1}, targets=['piFHres','PFHres'], solver="broyden_custom")
def nkpc_I(piFH, PFH, rinc, E, kappa_I):
    PFHres = PFH(-1) + piFH - PFH
    piFHres = kappa_I * (E/PFH - 1) + piFH(+1) / (1 + rinc) - piFH
    return piFHres, PFHres

@sj.solved(unknowns={'piHH': 0,'PHH': 1}, targets=['piHHres','PHHres'], solver="broyden_custom")
def nkpc(piHH, PHH, w, Z, P, rinc, markup_ss, kappa_p):
    real_mc = w * P / (Z * PHH)
    PHHres = PHH(-1) + piHH - PHH
    piHHres = piHH - piHH(+1)/(1 + rinc) - kappa_p * markup_ss * (real_mc - 1/markup_ss)
    return piHHres, PHHres, real_mc

@sj.solved(unknowns={'piHF': 0,'PHF': 1}, targets=['piHFres','PHFres'], solver="broyden_custom")
def nkpc_X(piHF, PHF, PHH, rinc, E, kappa_X):
    PHFres = PHF(-1) + piHF - PHF
    piHFres = kappa_X * (PHH/(E*PHF)-1) + piHF(+1) / (1 + rinc) - piHF
    return piHFres, PHFres

@sj.simple
def cpi(piHH,piFH,alpha):
    piout = (1-alpha)*piHH + alpha*piFH
    return piout

@sj.simple
def prices(P,PHH,PHF,PFH):
    phh = PHH/P
    pfh = PFH/P                  # price of foreign goods in home currency
    phf = PHF                    # price of home goods in foreign currency
    return phh, phf, pfh

@sj.simple
def eq_quant(pi, piout, piw, w, r, i):
    real_wage = piw - pi - (w - w(-1))
    pires = piout - pi
    fisher = r + piout(1) - i
    return real_wage, pires, fisher

""" Endogenous UIP """

@sj.simple
def rstar_eq(rstar, rstar_exo, phi_rstar, nfa):
    rstar_res = rstar - (rstar_exo - phi_rstar*nfa)
    return rstar_res

""" E shocks """

@sj.solved(unknowns={'E':1}, targets=['Eres'], solver="brentq")
def piE_to_E(piE, E):
    Eres = E - E(-1) - piE
    return Eres

@sj.simple
def monetary_Erule(ishock):
    piE = ishock
    return piE

@sj.simple
def UIP_Erule(E, piE, P, rstar):
    i = rstar + piE(1)
    Q = E/P
    return i, Q

""" Incomplete market RA model """

@sj.solved(unknowns={'C':1,'A':1}, targets=['Cres','Ares'], solver="broyden_custom")
def hh_RA(C, A, atw_n, rpost, rsub, beta, eis):
    Cres = (beta * (1 + rsub(1))) ** (-eis) * C(1) - C
    Ares = (1 + rpost) * A(-1) + atw_n - C - A
    UCE = C ** (-1/eis)
    return Cres, Ares, UCE

def phf_f(Q, eta, alpha):
    if eta == 1:
        phf = Q**(-1/(1-alpha))
    else: 
        phf = ((Q**(eta-1) - alpha)/(1-alpha))**(1/(1-eta))
    return phf

def phh_f(Q,eta, alpha):
    if eta == 1:
        phh = Q**(-alpha/(1-alpha))
    else: 
        phh = ((1-alpha*Q**(1-eta))/(1-alpha))**(1/(1-eta))
    return phh

def euler(Cnext,r,beta,sigma):
    return (beta*(1+r))**(-1/sigma)*Cnext

def uip(Qnext,r,rstar):
    return Qnext*(1+rstar)/(1+r)

def q_f(r_irf,T):
    disc = 1/(1+r_irf)
    q_irf = np.ones(T)
    for t in range(1,T): q_irf[t] = np.prod(disc[:t])
    return q_irf

def nfa_f(nfapast,rpast,phh,y,C):
    return (1+rpast)*nfapast + phh*y - C

def RA_solution(Q,y,r_irf,rstar_irf,ss,T):

    '''This solves the RA-IM model nonlinearly in the baseline model'''
    
    # Define parameters
    eta = ss['eta']
    alpha = ss['alpha']
    gamma = ss['gamma']
    eps_dcp = ss['eps_dcp']
    Cstar = ss['Cstar']
    beta = ss['beta']
    sigma = 1/ss['eis']
    r = ss['r']

    # Given Q, find C in the new steady state
    C = (y-alpha*phf_f(Q,eta,alpha)**(-gamma*eps_dcp)*Cstar)/((1-alpha)*phh_f(Q,eta,alpha)**(-eta))

    # Find path for C and Q backward
    C_irf = C*np.ones(T)
    Q_irf = Q*np.ones(T)
    for t in range(1,T):
        C_irf[T-t-1] = euler(C_irf[T-t],r_irf[T-t-1],beta,sigma)
        Q_irf[T-t-1] = uip(Q_irf[T-t],r_irf[T-t-1],rstar_irf[T-t-1])

    # Compute path for Y
    y_irf = (1-alpha)*phh_f(Q_irf,eta,alpha)**(-eta)*C_irf + alpha*phf_f(Q_irf,eta,alpha)**(-gamma*eps_dcp)*Cstar

    # Compute path for discount factor q
    q_irf = q_f(r_irf,T)

    # Compute residual in budget constraint
    res = q_irf@(C_irf-phh_f(Q_irf,eta,alpha)*y_irf) + q_irf[T-1]/r*(C - y*phh_f(Q,eta,alpha))

    # Compute nfa
    nfa_irf = np.zeros(T)
    nfa_irf[0] = phh_f(Q_irf[0],eta,alpha)*y_irf[0] - C_irf[0]
    for t in range(1,T): nfa_irf[t] = np.round(nfa_f(nfa_irf[t-1],r_irf[t-1],phh_f(Q_irf[t],eta,alpha),y_irf[t],C_irf[t]),decimals=10)

    # long run wage inflation
    piw = ss['kappa_w']/(1-ss['beta']) * (ss['vphi']*(y/ss['Z'])**(1/ss['frisch']) - phh_f(Q,eta,alpha)*ss['Z']*C**(-sigma)/(ss['markup_ss']*ss['markup_ss']))
        
    return res, C_irf, Q_irf, y_irf, nfa_irf, piw

""" Complete market HA model """

def CMHA_compute(chilist,dQ,T,ss):
    
    """Computes the response in model CM-HA"""

    # Compute transition matrix for policy and distribution on (a,e_{-1})
    Dlast = Dlast_compute(ss)
    
    # Get inputs from ssj
    h=1e-4
    M_het, mpc, D, outputs, differentiable_backward_fun, differentiable_hetinput, differentiable_hetoutput, law_of_motion, exog_by_output, curlyPs, ss0 = prelim_get(ss, T=T, h=h)
    
    # Compute Mt matrix: response of C at all t for a transfer at 0, which was in response to a shock at time s
    Mt = np.zeros((T,T))
    for t in range(T):

        # Define shock
        shock = np.zeros(T)
        shock[t] = h

        # Find transfer at t=0 such that consumption remains constant after shock
        dc = ((M_het.T)@shock).T
        T_het_last = -(ss['Pi']@(ss['c']**(-1/ss['eis']-1)*dc))/(ss['Pi']@(ss['c']**(-1/ss['eis']-1)*mpc))
        T_het = ss['Pi'].T@(T_het_last*Dlast)/D
        T_het[np.isnan(T_het)] = 0
        ss0['M'] = T_het                                                            # this is used in the next step

        # Compute curlyY and curlyD (backward iteration) - remember that those are response of Y and D TODAY to a shock at t
        curlyYs, curlyDs = hh_HA.backward_fakenews('Transfer', outputs, 2, differentiable_backward_fun,
                                                                      differentiable_hetinput, differentiable_hetoutput,
                                                                      law_of_motion, exog_by_output)

        # Compute F
        Tpost = curlyPs.shape[0] - T + 2
        F = curlyPs.reshape((Tpost + T - 2, -1)) @ curlyDs.reshape((2, -1)).T

        # Fill in Mt
        Mt[0,t] = curlyYs['c'][0]/h
        Mt[1:,t] = F[:,0]/h

    # Compute new M matrix incl. transfers
    M, Md, JCY = get_M(ss,T)
    M_withTf = M + Mt

    # Compute GE
    dY, dC = {}, {}
    for chi in chilist:
        dY[chi] = np.linalg.inv(np.identity(T) - (1-ss['alpha'])*M_withTf)@((ss['alpha']/(1-ss['alpha'])*chi*np.identity(T)-ss['alpha']*M_withTf)@dQ)
        dC[chi] = -ss['alpha']/(1-ss['alpha'])*M_withTf@dQ + M_withTf@dY[chi]
        
    return dY, dC, M_withTf

def backward_step_fakenews_het(self, din_dict, output_list, differentiable_backward_fun, law_of_motion: ForwardShockableTransition):
    
    """Gets response of consumption at every state at t to a shock to input at t"""

    # shock perturbs outputs
    shocked_outputs = differentiable_backward_fun.diff(din_dict)
    curlyV = {k: law_of_motion[0].expectation(shocked_outputs[k]) for k in self.backward}

    # and also affect aggregate outcomes today
    curlyY_het = {k: shocked_outputs[k] for k in output_list}

    return curlyV, curlyY_het

def backward_fakenews_het(self, input_shocked, output_list, T, differentiable_backward_fun,
                        differentiable_hetinput, law_of_motion: ForwardShockableTransition):
    
    """Gets response of consumption at every state at t to a shock to input at t"""
    
    # contemporaneous effect of unit scalar shock to input_shocked
    din_dict = {input_shocked: 1}
    if differentiable_hetinput is not None and input_shocked in differentiable_hetinput.inputs:
        din_dict.update(differentiable_hetinput.diff({input_shocked: 1}))
    curlyV, curlyY_het = backward_step_fakenews_het(self, din_dict, output_list, differentiable_backward_fun, law_of_motion)

    # infer dimensions from this, initialize empty arrays, and fill in contemporaneous effect
    curlyY_hets = {k: np.empty((T,) + curlyY_het[k].shape) for k in curlyY_het.keys()}
    for k in curlyY_het.keys():
        curlyY_hets[k][0] = curlyY_het[k]

    # fill in anticipation effects of shock up to horizon T
    for t in range(1, T):
        curlyV, curlyY_het = backward_step_fakenews_het(self, {k+'_p': v for k, v in curlyV.items()},
                                                output_list, differentiable_backward_fun, law_of_motion)
        for k in curlyY_het.keys():
            curlyY_hets[k][t] = curlyY_het[k]

    return curlyY_hets

def Dlast_compute(ss):
    
    """Computes the distribution on (a,e_{-1})"""
    
    # Get coordinates for savings policy
    ida, w = interpolate.interpolate_coord(ss['a_grid'], ss['a'])

    # Fill in new distribution in terms of (e,a')
    D = ss.internals['hh_HA']['D']
    Dlast = np.zeros_like(D)
    for ie in range(ss['n_exog']):
        for ia in range(ss['nA']):
            Dlast[ie,ida[ie,ia]] += D[ie,ia]*w[ie,ia]
            Dlast[ie,ida[ie,ia]+1] += D[ie,ia]*(1-w[ie,ia])

    return Dlast

def transition_compute(ss):
    
    """Computes law of motion from (a_{-1),e_{-1}) to (a,e)"""
    
    # Get coordinates for savings policy
    ida, w = interpolate.interpolate_coord(ss['a_grid'], ss['a'])

    # Fill in transition matrix using savings policy and exongeous transition probability
    Transition = np.zeros((ss['n_exog']*ss['nA'],ss['n_exog']*ss['nA']))
    for ie in range(ss['n_exog']):
        for ienext in range(ss['n_exog']):
            for ia in range(ss['nA']):
                Transition[ia+ie*ss['nA'],ida[ie,ia]+ienext*ss['nA']] = ss['Pi'][ie,ienext]*w[ie,ia]
                Transition[ia+ie*ss['nA'],ida[ie,ia]+1+ienext*ss['nA']] = ss['Pi'][ie,ienext]*(1-w[ie,ia])

    # Multiply transition matrix by stationary distribution to get actual masses getting into each (a,e')
    Dss = np.reshape(ss.internals['hh_HA']['D'],(ss['n_exog']*ss['nA']))
    Transition = (Dss*Transition)

    # Normalize transition matrix so that columns sum to 1
    Transition = Transition/np.sum(Transition,0)
    Transition[np.isnan(Transition)] = 0

    return Transition, ida, w

def prelim_get(ss, T=400, h=1e-4):

    """Computes the M matrix for each state (a,e) and other inputs"""
    
    # Define HA block
    self = hh_HA

    # Initialize some inputs that we need for the Fake News Algorithm
    ss0 = self.extract_ss_dict(ss)
    outputs = self.M_outputs.inv @ ['c']
    exog = self.make_exog_law_of_motion(ss0)
    endog = self.make_endog_law_of_motion(ss0)
    differentiable_backward_fun, differentiable_hetinput, differentiable_hetoutput = self.jac_backward_prelim(ss0, h, exog, False)
    law_of_motion = CombinedTransition([exog, endog]).forward_shockable(ss0['Dbeg'])
    exog_by_output = {k: exog.expectation_shockable(ss0[k]) for k in outputs | self.backward}
    curlyPs = self.expectation_vectors(ss0['c'], T-1, law_of_motion)

    # Compute Curly Y (response of consumption to inputs) at the individual level
    curlyY_hets_w = backward_fakenews_het(self, 'n', ['c'], T, differentiable_backward_fun, differentiable_hetinput, law_of_motion)
    curlyY_hets_r = backward_fakenews_het(self, 'rpost', ['c'], T, differentiable_backward_fun, differentiable_hetinput, law_of_motion)
    curlyY_hets_Tf = backward_fakenews_het(self, 'cbarF', ['c'], T, differentiable_backward_fun, differentiable_hetinput, law_of_motion)

    # Construct M at the individual level (response of consumption to real income at t)
    Jd = 1/ss['j']*(1/(1+ss['r']))**np.arange(T)
    Md = (Jd[:,np.newaxis,np.newaxis]*curlyY_hets_r['c'][0,:,:])
    M_het = (1-1/ss['markup_ss'])*Md + curlyY_hets_w['c']               # note: this is the Jacobian with respect to w so markups do not show up

    # MPC (response of consumption to transfer at 0)
    mpc = -curlyY_hets_Tf['c'][0]
    
    # Stationary distribution
    D = ss0['D']

    return M_het, mpc, D, outputs, differentiable_backward_fun, differentiable_hetinput, differentiable_hetoutput, law_of_motion, exog_by_output, curlyPs, ss0

def transfer_compute_eq(dQ,dY,ss):
    
    """Computes transfer for each (a,e) given equilibrium dQ and dY"""

    # Get inputs from ssj
    M_het, mpc, D, _, _, _, _, _, _, _, _ = prelim_get(ss, T=len(dQ), h=1e-4)

    # Compute transition matrix for policy
    _, ida, w = transition_compute(ss)

    # Compute change in individual consumption at t=0
    dc = -ss['alpha']/(1-ss['alpha'])*((M_het.T)@dQ).T + ((M_het.T)@dY).T
    
    # Compute transfer
    T_het_last = -(ss['Pi']@(ss['c']**(-1/ss['eis']-1)*dc))/(ss['Pi']@(ss['c']**(-1/ss['eis']-1)*mpc))

    # Compute transfer on the (a0,e)
    T_het_last2 = np.zeros((ss['n_exog'],ss['nA']))
    for ie in range(ss['n_exog']):
        for ia in range(ss['nA']):
            T_het_last2[ie,ia] = w[ie,ia]*T_het_last[ie,ida[ie,ia]] + (1-w[ie,ia])*T_het_last[ie,ida[ie,ia]+1]

    return T_het_last, dc
