# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:31:21 2017

@author: nickilla
"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.interpolate as interpolate
from numba import jit, njit, prange
from joblib import Parallel, delayed


class MultiRegionLearning:
    '''
    Class that stores parameters and solution methods for the multiregion
    learning model from Garnadt (2017).
    '''
    
    def __init__(self,
                 L = np.array((1.0,2.0)),       #Population size of the regions
                 D = np.array(((1.0,1.5),
                               (1.5,1.0))),   #Distance matrix
                 gamma = lambda x: x**0.5,  #Distance -> coordination cost
                 xmin=1.0,                    #scale of the productivity distribution
                 alpha=8.0,                   #shape of the productivity distribution 
                 mu_theta=0.0,                #mean of the preference parameters
                 sig_theta=1.0,               #variance of the preference parameters
                 sig_eps=1.0,                 #variance of the preference shock
                 f_h=1.0,                     #fixed cost of producing in HQ market
                 f_f=1.5,                   #fixed cost of producing in non-HQ market
                 f_e=5.0,                     #market entry cost
                 delta=0.02,                #death shock probability
                 nmax=150,                  #maximum age of a firm
                 sigma=6.0,                   #CES preference parameter
                 beta=0.96,                 #Time discount rate
                 hermord = 21):             #Nr grid points for Gauss-Hermite
        self.L, self.D, self.gamma = L, D, gamma
        self.xmin, self.alpha = xmin, alpha
        self.phi_dist = stats.pareto(alpha,scale=xmin)
        self.mu_theta, self.sig_theta = mu_theta,sig_theta
        self.theta_dist = stats.norm(loc=mu_theta,scale=sig_theta**2)
        self.sig_eps = sig_eps
        self.eps_dist = stats.norm(scale=sig_eps**2)
        self.adist = stats.norm(loc=mu_theta,scale=sig_eps**2+sig_theta**2)
        self.f_h, self.f_f, self.f_e = f_h, f_f, f_e
        self.delta, self.nmax, self.sigma, self.beta = delta, nmax, sigma, beta
        self.agegrid = np.array(range(nmax))
        self.phigrid = np.linspace(self.phi_dist.ppf(0),self.phi_dist.ppf(0.9999),25)
        self.shockgrid = np.linspace(self.adist.ppf(0.001),self.adist.ppf(0.999),25)
        self.ghpoints, self.ghweights = np.polynomial.hermite.hermgauss(hermord)
    
    @jit
    def mu(self,abar,n):
        '''
        Expected value of next periods demand shifter when the firm has observed
        n signals and their mean was abar
        '''
        mu_theta, sig_theta, sig_eps = self.mu_theta, self.sig_theta, self.sig_eps
        return (mu_theta*sig_eps**2+n*abar*sig_theta**2)/(n*sig_theta**2+sig_eps**2)
    
    @jit       
    def nu(self,n):
        '''
        Variance of the belief about next periods demand shifter when the firm 
        has observed n signals
        '''
        sig_theta, sig_eps = self.sig_theta, self.sig_eps
        return (sig_theta**2 * sig_eps**2)/(n*sig_theta**2 + sig_eps**2) + sig_eps**2
    
    @jit
    def normpdf(self,x,mu,sigma_2):
        '''
        pdf at value x of a normal distribution with mean mu and variance
        sigma_2
        '''
        val = 1/np.sqrt(2*np.pi*sigma_2)*np.exp(-(x-mu)**2/(2*sigma_2))
        return val
    
    @jit(nopython=True)
    def paretopdf(self,x,scale,shape):
        '''
        pdf at value x of a pareto distribution with scale and shape
        '''
        if x < scale:
            val = 0
        else:
            val = shape*scale**shape/x**(shape+1)
            
        return val
    
    @jit
    def linterp(self,x: np.ndarray, xp: np.ndarray, fp: np.ndarray, result: np.ndarray):
        """Similar to numpy.interp, if x is an array. Also preallocate the 
        result vector as this doubles the speed"""
        M = len(x)
        
        for i in range(M):
            i_r = np.searchsorted(xp, x[i])
            
            # These edge return values are set with left= and right= in np.interp.
            if i_r == 0:
                interp_port = (x[i] - xp[i_r]) / (xp[i_r+1] - xp[i_r])
                result[i] = fp[i_r] + (interp_port * (fp[i_r+1] - fp[i_r]))
                continue
            elif i_r == len(xp):
                interp_port = (x[i] - xp[i_r-1]) / (xp[i_r] - xp[i_r-1])
                result[i] = fp[i_r-1] + (interp_port * (fp[i_r] - fp[i_r-1]))
                continue
            
            interp_port = (x[i] - xp[i_r-1]) / (xp[i_r] - xp[i_r-1])
            
            result[i] = fp[i_r-1] + (interp_port * (fp[i_r] - fp[i_r-1]))

        return result
    
    @jit    
    def expected_demand(self,abar,n):
        '''
        Expected value of the demand shifter exp(a/sigma) when the firm has
        observed n signals with mean abar
        '''
        mu = self.mu(abar,n)
        nu = self.nu(n)
        sigma, sig_eps = self.sigma, self.sig_eps
        return np.exp(mu/sigma+(nu)/sigma**2)
   
    @jit 
    def payoff(self,wage,P,homeloc,prodloc,phi,abar,n):
        '''
        Expected payoff of a firm with headquarters in homeloc with productivity
        phi, that has observed n demand shocks with average abar, when producing
        in prodloc. Payoff takes the vector of price levels P and the vector of
        wages wage as given. Notice that due to free entry the average firm
        profits are zero.
        '''
        sigma = self.sigma
        D = self.D
        gamma=self.gamma
        expected_demand=self.expected_demand
        Y = wage[prodloc]*self.L[prodloc]
        if homeloc == prodloc:
            fixed_cost = self.f_h
        else:
            fixed_cost = self.f_f
        
        payoff = (Y/P[prodloc]**(1-sigma)*(phi/
                  (gamma(D[homeloc,prodloc]*wage[prodloc])))**(sigma-1)*
                     ((sigma-1)/sigma * expected_demand(abar,n))**sigma*
                     1/(sigma-1)-fixed_cost*wage[prodloc])
        return payoff
    
    def bellman_operator(self,W,wage,P):
        '''
        Bellman operator for a given vector of location specific wages w and
        price levels P. Takes the current value function guess W and maps it
        into a new value function V. We assume that there is no more learning
        after len(ngrid) periods
        '''
        L = self.L
        beta = self.beta
        delta = self.delta
        payoff = self.payoff
        mu=self.mu
        nu=self.nu
        sig_eps=self.sig_eps
        V=np.empty_like(W)
        
        for i in range(len(L)):
            for phi_id, phi in enumerate(self.phigrid):
                for a_id, a in enumerate(self.shockgrid):
                    for n in self.agegrid:
                        
                        if n < len(self.agegrid)-2:
                            W_fun = lambda x: np.interp(x,self.shockgrid,W[i,phi_id,:,n+1])
                        else:
                            W_fun = lambda x: 0*x
                            
                        Vpayhome = payoff(wage,P,i,i,phi,a,n)
                        Vpayfor = sum(max(payoff(wage,P,i,j,phi,a,n),0)
                                    for j in range(len(L)))-max(Vpayhome,0)
                        
                        abardist = stats.norm(mu(a,n),nu(n)+sig_eps**2)
                        integrand = lambda x: W_fun(x)*abardist.pdf(x)
                        Vexp = beta*(1-delta)*integrate.quad(integrand,-np.inf,np.inf)[0]
                        
                        V[i,phi_id,a_id,n] = max(Vpayhome+Vpayfor+Vexp,0)
    
    
    def backwards_operator(self,wage: np.ndarray, P: np.ndarray,
                           W: np.ndarray, wval: np.ndarray, resvec: np.ndarray):
        '''
        Function that calculates the value function using backwards iteration 
        under the assumption that firms die after 150 years
        '''
        L = self.L
        beta = self.beta
        delta = self.delta
        payoff = self.payoff
        phigrid = self.phigrid
        shockgrid=self.shockgrid
        agegrid=self.agegrid
        lena=len(agegrid)
        mu=self.mu
        nu=self.nu
        Vpayhome = 0
        Vpayfor = 0
        
        
        
        #Initialize points and weights for Gauss hermite quadrature
        ghpoints = self.ghpoints
        ghweights = self.ghweights
        
        #Precalculate factors that are repeatedly used
        ghfactor = 1/np.sqrt(np.pi)
        sqrt2=np.sqrt(2)
        sqrtnun=np.sqrt(nu(agegrid))
        
        
        for i in range(len(L)):
            for n in self.agegrid:
                sqrtnuncur = sqrtnun[lena-n-1]
                for a_id, a in enumerate(shockgrid):
                    Vpayhome = payoff(wage,P,i,i,phigrid,a,lena-n-1)
                        
                    Vpayfor = 0
                    for j in range(len(L)):
                        Vpayfor +=np.maximum(payoff(wage,P,i,j,phigrid,a,lena-n-1),0)
                        
                    Vpayfor += -np.maximum(Vpayhome,0)
                        
                    mucur=mu(a,lena-n-1)
                        
                    if n > 0:
                        for phi_id in range(len(phigrid)):
                            wval[phi_id] = self.linterp(ghpoints*sqrt2*sqrtnuncur/(n+1)+(mucur+n*a)/(n+1),
                                                shockgrid,
                                                W[i,phi_id,:,lena-n],resvec)
                    else:
                        wval =  0*wval
                                                                               
                    Vexp = beta*(1-delta)*ghfactor*wval@ghweights
                    W[i,:,a_id,lena-n-1] = np.maximum(Vpayhome+Vpayfor+Vexp,0)
                        
        return W
    
    def EVenter(self,P,wage):
        '''
        Calculates the expected value of entering in each location given a wage
        and price level guess
        '''
        
        L=self.L
        shockgrid=self.shockgrid
        agegrid=self.agegrid
        phigrid=self.phigrid
        f_e=self.f_e
        
        Entvalue = np.ones(len(L))
        V = np.zeros((len(L),len(phigrid),len(shockgrid),len(agegrid)))
        wval = np.zeros((len(phigrid),len(self.ghpoints)))
        resvec = np.zeros(len(self.ghpoints))
        
        V = self.backwards_operator(wage,P,V,wval,resvec)
        #Draw simsize productivities from the productivity distribution
        #and calculate the value of entry:
        
        
        simsize = 10000000
        np.random.seed(123987)
        philist = self.phi_dist.rvs(simsize)
        for i in range(len(L)):
            Vint = interpolate.interp1d(phigrid,V[i,:,12,0],kind='cubic',
                                        fill_value=(V[i,0,12,0],V[i,len(phigrid)-1,12,0]),
                                        bounds_error=False)
            Entvalue[i] = sum(Vint(philist))/simsize-f_e*wage[i]
            
        return Entvalue
    
    @jit
    def find_phi_cutoffs(self,V):
        '''
        Finds the cutoff productivity levels phi*(i,abar,n) below which firms
        exit the market given a value function V
        '''
        L=self.L
        shockgrid=self.shockgrid
        agegrid=self.agegrid
        phigrid=self.phigrid
        
        cutoff = np.empty((len(L),len(shockgrid),len(agegrid)))
        for i in range(len(L)):
            for a_id, a in enumerate(shockgrid):
                for n in agegrid:
                    cutoff[i,a_id,n] = phigrid[np.searchsorted(V[i,:,a_id,n],1e-5)]
        
        return cutoff
    
    @jit
    def find_a_cutoffs(self,V):
        '''
        Finds the cutoff average observed shock levels abar*(i,phi,n) below
        which firms exit the market given a value function V
        '''
        L=self.L
        shockgrid=self.shockgrid
        agegrid=self.agegrid
        phigrid=self.phigrid
        
        cutoff = np.empty((len(L),len(phigrid),len(agegrid)))
        for i in range(len(L)):
            for phi_id, phi in enumerate(phigrid):
                for n in agegrid:
                    cutoff[i,phi_id,n] = shockgrid[np.searchsorted(V[i,phi_id,:,n],1e-5)]
                    
        return cutoff
    

    def generate_firm_density(self,V,M,mval,resvec):
        '''
        Generates the firm density over locations, productivity, average
        observed signals and age
        '''
        L = self.L
        phigrid = self.phigrid
        shockgrid = self.shockgrid
        agegrid = self.agegrid
        alpha = self.alpha
        xmin = self.xmin
        delta = self.delta
        nu=self.nu
        sig_eps=self.sig_eps
        sig_theta=self.sig_theta
        mu_theta=self.mu_theta
        
        m = np.zeros((len(L),len(phigrid),len(shockgrid),len(agegrid)))
        phi_cutoff = self.find_phi_cutoffs(V)
        a_cutoff = self.find_a_cutoffs(V)
        
        ghpoints = self.ghpoints
        ghweights= self.ghweights
        sqrtpi=np.sqrt(np.pi)
        
        for i in range(len(L)):
            for n in agegrid:
                
                curfact = (sig_eps**2)/(n*sig_theta**2+sig_eps**2)
                if n==0:
                    nulastfact = 0
                else:
                    nulastfact=np.sqrt(2*nu(n-1))
                    
                for a_id, a in enumerate(shockgrid):
                    
                        if n == 0:
                            phi_act = phigrid > phi_cutoff[i,a_id,n]
                            if a==0:
                                m[i,:,a_id,n] = M[i]*alpha*xmin**alpha/(phigrid**(alpha+1))*phi_act
                            else:
                                m[i,:,a_id,n] = 0
                        elif n==1:
                            #Numerical integration delivers bad results for
                            #for mass points. Thus use the fact that the first
                            #periods distribution is just a truncated normal
                            #around mu_theta multiplied by the pareto pdf of the
                            phi_act = phigrid > phi_cutoff[i,12,0]
                            #productivity draw
                            indicator = phigrid>=phi_cutoff[i,a_id,n]
                            m[i,:,a_id,n]= M[i]*alpha*xmin**alpha/(phigrid**(alpha+1))* \
                                                phi_act*1/sqrtpi*1/nulastfact*\
                                                np.exp(-(a-mu_theta)**2/nu(n-1))*indicator
                        else:
                            #Use Gauss-Hermite quadrature to calculate the
                            #integral over past periods average signals
                            #Current periods mass function is the integral over
                            #past periods mass for each average signal a(-1) times 
                            #the probability of the shock that generates the
                            #new average signal a given that the past average
                            #signal was a(-1)
                            
                            xval = (ghpoints*nulastfact-(n+1)*a+mu_theta* \
                                    curfact)/(n+1 - curfact)
                            
                            for phi_id in range(len(phigrid)):
                                indicator = xval >= a_cutoff[i,phi_id,n-1]
                                mval[phi_id] = self.linterp(xval,shockgrid,m[i,phi_id,:,n-1],resvec)*indicator
                                
                            m[i,:,a_id,n] = (1-delta)*(n+1)/(n+1-curfact)*1/sqrtpi*mval@ghweights
                            
        return m
                            
        
        
    
    def inner_loop(self,wage):
        '''
        For a given verctor of wages wage, this function finds the vector of 
        price levels P and masses of entering firms M such that the goods
        market clears.
        '''
        L=self.L
        shockgrid=self.shockgrid
        agegrid=self.agegrid
        phigrid=self.phigrid
        
        P = 1.5/L
        M = np.ones(len(L))
        V = np.ones((len(L),len(phigrid),len(shockgrid),len(agegrid)))
        m = np.empty_like(V)
        Entvalue = np.ones(len(L))
        
        Optout = optimize.root(self.EVenter,P,args=(wage),method='lm',options={'maxiter' : 30})
        Pout = Optout['x']
        print(Pout)
        
        V = np.zeros((len(L),len(phigrid),len(shockgrid),len(agegrid)))
        wval = np.zeros((len(phigrid),len(self.ghpoints)))
        resvec = np.zeros(len(self.ghpoints))
        V = self.backwards_operator(wage,Pout,V,wval,resvec)
        
        simsize = 10000000
        philist = self.phi_dist.rvs(simsize)
        for i in range(len(L)):
            Vint = interpolate.interp1d(phigrid,V[i,:,12,0],kind='cubic',
                                        fill_value=(V[i,0,12,0],V[i,len(phigrid)-1,12,0]),
                                        bounds_error=False)
            Entvalue[i] = sum(Vint(philist))/simsize-self.f_e*wage[i]
        
        #Calculate the distribution over productivity, age and average signals
        mval = np.zeros((len(phigrid),len(self.ghpoints)))
        resvec = np.zeros(len(self.ghpoints))
        m = self.generate_firm_density(V,M,mval,resvec)
            
        #Calculate the mass of entrants that generates the correct price level
        
        #Calculate labor demand in each location
            
        return Pout, M, V, m, Entvalue
            
            
        
        
        