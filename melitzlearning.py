# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:31:21 2017

@author: nickilla
"""
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
from numba import jit


class MultiRegionLearning:
    '''
    Class that stores parameters and solution methods for the multiregion
    learning model from Garnadt (2017).
    '''
    
    def __init__(self,
                 L = np.array((1,2)),       #Population size of the regions
                 D = np.array(((1,1.5),
                               (1.5,1))),   #Distance matrix
                 gamma = lambda x: x**0.5,  #Distance -> coordination cost
                 xmin=1,                    #scale of the productivity distribution
                 alpha=3,                   #shape of the productivity distribution 
                 mu_theta=0,                #mean of the preference parameters
                 sig_theta=1,               #variance of the preference parameters
                 sig_eps=1,                 #variance of the preference shock
                 f_h=1,                     #fixed cost of producing in HQ market
                 f_f=1.5,                   #fixed cost of producing in non-HQ market
                 f_e=5,                     #market entry cost
                 delta=0.02,                #death shock probability
                 nmax=150,                  #maximum age of a firm
                 sigma=6,                   #CES preference parameter
                 beta=0.96):                #Time discount rate
        self.L, self.D, self.gamma = L, D, gamma
        self.xmin, self.alpha = xmin, alpha
        self.phi_dist = stats.pareto(xmin,scale=alpha)
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
        Variance of next periods demand shifter when the firm has observed
        n signals
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
    
    @jit
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
    def linterp(self,x,xval,yval):
        '''
        linear interpolator that can be jitted by numba
        '''
        pos = np.searchsorted(xval,x)
        if pos == 0:
            pos = 1
        elif pos == len(xval):
            pos = len(xval)-1

        a = yval[pos-1]
        b = (yval[pos]-yval[pos-1])/(xval[pos]-xval[pos-1])
        intval = a+b*(x-xval[pos-1])
        
        return intval
    
    @jit    
    def expected_demand(self,abar,n):
        '''
        Expected value of the demand shifter exp(a/sigma) when the firm has
        observed n signals with mean abar
        '''
        mu = self.mu(abar,n)
        nu = self.nu(n)
        sigma, sig_eps = self.sigma, self.sig_eps
        return np.exp(mu/sigma+(nu+sig_eps**2)/sigma**2)
   
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
        into a new value function V.
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
    

    def backwards_operator(self,W,wage,P):
        '''
        Bellman operator for a given vector of location specific wages w and
        price levels P. Takes the current value function guess W and maps it
        into a new value function V.
        '''
        L = self.L
        beta = self.beta
        delta = self.delta
        payoff = self.payoff
        phigrid = self.phigrid
        shockgrid=self.shockgrid
        mu=self.mu
        nu=self.nu
        sig_eps=self.sig_eps
        
        for i in range(len(L)):
            for phi_id, phi in enumerate(phigrid):
                for n in self.agegrid:
                    for a_id, a in enumerate(shockgrid):
                                            
                        Vpayhome = payoff(wage,P,i,i,phi,a,len(self.agegrid)-n-1)
                        Vpayfor = sum(max(payoff(wage,P,i,j,phi,a,len(self.agegrid)-n-1),0)
                                    for j in range(len(L)))-max(Vpayhome,0)
                        
                        @jit
                        def integrand(x):
                            if n > 0:
                                wval = self.linterp(x,shockgrid,W[i,phi_id,:,len(self.agegrid)-n])
                            else:
                                wval =  0
                                                        
                            val = wval*self.normpdf(x,mu(a,n),nu(n)+sig_eps**2)
                            return val
                        
                        Vexp = beta*(1-delta)*integrate.quad(integrand,-np.inf,np.inf)[0]
                        W[i,phi_id,a_id,len(self.agegrid)-n-1] = max(Vpayhome+Vpayfor+Vexp,0)
                        
        return W
    
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
                    cutoff[i,a_id,n] = phigrid[np.searchsorted(V[i,:,a_id,n],0)]
        
        return cutoff
    
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
                    cutoff[i,phi_id,n] = shockgrid[np.searchsorted(V[i,phi_id,:,n],0)]
                    
        return cutoff
    
    def generate_firm_density(self,V,M):
        '''
        Generates the firm density over locations, productivity, average
        observed signals and age
        '''
        L = self.L
        phigrid = self.phigrid
        shockgrid = self.shockgrid
        agegrid = self.agegrid
        phi_dist = self.phi_dist
        delta = self.delta
        mu=self.mu
        nu=self.nu
        sig_eps=self.sig_eps
        
        m = np.zeros((len(L),len(phigrid),len(shockgrid),len(agegrid)))
        phi_cutoff = self.find_phi_cutoffs(V)
        a_cutoff = self.find_a_cutoffs(V)
        
        for i in range(len(L)):
            for phi_id, phi in enumerate(phigrid):
                for a_id, a in enumerate(shockgrid):
                    for n in agegrid:
                        if n == 0:
                            phi_act = phi > phi_cutoff[i,a_id,n]
                            m[i,phi_id,a_id,n] = M[i]*phi_dist.pdf(phi)*phi_act
                        else:
                            m_fun = lambda x: np.interp(x,shockgrid,m[i,phi,:,n-1])
                            abardist = lambda x: stats.norm(mu(x,n-1),nu(n-1)+sig_eps**2)
                            integrand = lambda x: (1-delta)*m_fun(x)*abardist(x).pdf(a)
                            m[i,phi_id,a_id,n] = integrate.quad(integrand,a_cutoff[i,phi_id,n-1],np.inf)[0]
                            
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
        
        P = np.ones(len(L))
        M = np.ones(len(L))
        V = np.ones((len(L),len(phigrid),len(shockgrid),len(agegrid)))
        m = np.empty_like(V)
        
        error_tol = 1e-4
        iter_max = 500
        error = 1
        count = 0
        
        while error > error_tol and count < iter_max:
            count +=1
            V = self.backwards_operator(V,wage,P)
            m = self.generate_firm_density(V,M)
            
        return P, M, V, m
            
            
        
        
        