# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:43:53 2021

@author: filipavaldeira
"""

import numpy as np
from scipy.optimize import root_scalar
from itertools import starmap

##############################################################################
##### Aux functions
##############################################################################

def func_LSE(x):
    # lse = np.log(1+np.exp(x))
    # For numerical stability
    a = x.max()
    lse = a + np.log(np.exp(-a) + np.exp(x-a))
    return lse

def cost_approx(A,w, x, gamma_reg = 1e-5):
    f = np.sum( func_LSE(1- A@x) *w) +gamma_reg*np.linalg.norm(x)
    return f    

def norm_diff_norm(x, xold):
    return np.linalg.norm((xold-x))/np.linalg.norm(xold)

def func_g(u,*data):
    x,lbda = data
    
    g = -np.exp(1-u)/(1+np.exp(1-u))+(u-x)/lbda
    return g

def dg_du(u,*data):
    """ implementation first derivative of g
        -
    """ 
    x,lbda = data
    return np.exp(1-u)/(1+np.exp(1-u))**2+1/lbda

def d2g_du2(u,*data):
    """ implementation second derivative of g
        -
    """ 
    return np.exp(u+1)*(np.exp(1)-np.exp(u))/(np.exp(1)+np.exp(u))**3


def df_val(x,w,v):
    return -np.exp(1-x)/(np.exp(1-x)+1)+(1/(w))*(x-v)

##############################################################################
##### PDRank
##############################################################################


class PDRANK(object):

    def __init__(self, rho = 1.9, step_ratio = 1, reg_gamma = 1e-4,  epsilon = 1e-2,
                 in_MAXITER= int(500), out_MAXITER = 25, epsilon_in = 1e-2, epsilon_out = 1e-2):
        
        """ PdRank algorithm for solving a rank aggregation problem.

            Parameters
            ----------    
                       
            rho : float
                step size for PDRank            
            step_ratio : float
                ratio for primal and dual step in PDRank
            reg_gamma : float
                regularization parameter (gamma in paper)
            epsilon : float
                epsilon for updating weights. Epsilon in paper. Recommended 1e-2.
            out_MAXITER : int
                maximum number of iterations for outer cycle                
            in_MAXITER : int
                maximum number of iterations for inner cycle                               
            epsilon_in : float
                stopping criterion for inner cycle
            epsilon_out : float
                stopping criterion for outer cycle
            ----------
        """ 
     
        self.rho = rho
        self.step_ratio = step_ratio
        self.out_MAXITER = int(out_MAXITER)
        self.in_MAXITER = int(in_MAXITER)
        self.epsilon_in = epsilon_in
        self.epsilon = epsilon
        self.epsilon_out = epsilon_out
        self.reg_gamma = reg_gamma

        # store intermediate solutions
        self.sv_w, self.sv_x, self.sv_x_out, self.sv_w_inner= [], [], [] , []

    def get_cost(self):
        cost = [cost_approx(self.A, w*self.w_obs, x, self.reg_gamma) for w, x in zip(self.sv_w_inner, self.sv_x) ]
        return cost


    def init_vars(self, A, m, n):
        """
        Initializes the variables x, w, and z.
        
        Parameters:
            A (numpy.ndarray): A.
            m (int): number items.
            n (int): number of comparisons.
        
        Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): A randomly initialized vector of size m, normalized to sum to 1.
            - w (numpy.ndarray): A vector of ones of size n.
            - z (numpy.ndarray): The result of the dot product of A and x.
        """

        x = np.random.randn(m).ravel()
        x = x/sum(x)
        z = A.dot(x)
        w = np.ones(n)
        return x, w, z
    
    
    def stop_criterion_inner_epsilon(self, cost_val, cost_old):
        criterion = np.abs(cost_val-cost_old)/cost_old
        return criterion
    
    def stop_criterion_inner(self, A, w, w_obs, x, x_old):    
        """
        Determines whether the stopping criterion for the inner loop is met.
        Stopping criterion is met if the relative change in the cost function is smaller than epsilon or the current cost is zero.
        
        Parameters:
            A (numpy.ndarray): A matrix.
            w (numpy.ndarray): Weights.
            w_obs (numpy.ndarray): Weight observations.
            x (numpy.ndarray): Current solution vector.
            x_old (numpy.ndarray): Previous solution vector.

        Returns:
            bool: True if the stopping criterion is met, False otherwise.
        """

        cost_old = cost_approx(A,w*w_obs,x_old,self.reg_gamma) 
        cost_val = cost_approx(A,w*w_obs,x,self.reg_gamma)
        return (self.stop_criterion_inner_epsilon(cost_val, cost_old) < self.epsilon_in) | (cost_val==0) 
       
    def stop_criterion_outer(self, sv_w, sv_x_out, eps_stop ,i_out):
        """
        Determines whether the stopping criterion for the outer loop is met.
        Criterion is met if either the weights vector or the solution vector 
        has a different to the previous iteration smaller than esp_stop.
        
        Parameters:
            sv_w (list): A list of weight vectors from previous iterations.
            sv_x_out (list): A list of solutions from previous iterations.
            eps_stop (float): The threshold for the stopping criterion.
            i_out (int): The current iteration index of the outer loop.
        
        Returns:
            bool: True if the stopping criterion is met, False otherwise.
        """

        if i_out> 1:
            w_diff = norm_diff_norm(sv_w[-1],sv_w[-2])
            x_diff = norm_diff_norm(sv_x_out[-1], sv_x_out[-2])
            stop = (w_diff<eps_stop)|(x_diff<eps_stop)
        else:
            stop = False
        return stop
    

    def proxTauF(self,u):
        """
        Proximal operator for function f
        
        """

        return (u - np.mean(u)).ravel()
    
    def proxTauG(self,ws,tau,vs,xinit=None):
        """
        Proximal operator for function g
        
        """      
        w_bar = ws*tau
        if xinit is None:
            res = np.fromiter(starmap(self.proxTauGi, zip(w_bar,vs)),dtype='float')
        else:
            res =np.fromiter(starmap(self.proxTauGi, zip(w_bar,vs,xinit)),dtype='float')
        return res

    def proxTauGi(self, w, v, xinit=None):
        if xinit is None:
            xinit = v 
        data = (v,w)
        a, b = -15,15
        if df_val(a,w,v)>0:
            sol = w+v
        elif df_val(b,w,v)<0:
            sol = v
        else:            
            bracket = [max(a,v),b]
            sol_root = root_scalar(func_g,args = data,bracket=bracket,x0=xinit,fprime=dg_du, fprime2=d2g_du2)
            sol = sol_root.root            
        return sol
    
    def update_w(self, A, x, eps):
        return  1/( np.log(1+np.exp(1-A@x))+ eps)
    
    def get_tau_sigma(self, A):
        tau = 1/(2*A.shape[0])**0.5
        tau_PDGH, sigma_PDGH = tau, tau
        return tau_PDGH, sigma_PDGH
    
    def inner_steps(self, A, z, x, reg_term, tau_PDGH, sigma_PDGH, step_size, w, w_obs, proxG_sol):
        
        x_bar = self.proxTauF(reg_term * (x-tau_PDGH*(A.T).dot(z)) )                
        u = z + sigma_PDGH * A.dot((2*x_bar - x))                
        proxG_sol = self.proxTauG(w*w_obs, 1/sigma_PDGH, u/sigma_PDGH, xinit = proxG_sol)
        z_bar = u - sigma_PDGH* proxG_sol 
        x, z = x + step_size*(x_bar - x), z + step_size*(z_bar - z)        

        return x, z, proxG_sol

    def solve(self, A, w_obs = None):
        """
        Solves the optimization problem.
        
        Parameters:
            A (numpy.ndarray): The input matrix of shape (n, m). Where each row corresponds to a pairwise comparison.
            w_obs (numpy.ndarray, optional): The number of times each pair was observed (length n). Defaults to None.
        
        Returns:
            numpy.ndarray: The solution vector x of shape (m,).

        """


        n, m = A.shape
        x, w, z = self.init_vars( A, m, n)
        if w_obs is None : w_obs = np.ones(n)

        tau_PDGH, sigma_PDGH = self.get_tau_sigma(A)      

        proxG_sol = np.ones(z.shape)
        step_size = self.rho
        reg_term = (1/(1+2*self.reg_gamma)) # regularization term

        # outer iterations
        for i_out in range(self.out_MAXITER):
            
            if i_out > 0:
                w = self.update_w(A,x, self.epsilon)

            self.sv_w.append(w)
            
            # inner cycle
            for iter in range(self.in_MAXITER):                
                
                # update x and z, keep x previous iteration
                xold = x
                x, z, proxG_sol = self.inner_steps( A, z, x, reg_term, tau_PDGH, sigma_PDGH, step_size, w, w_obs, proxG_sol)                              
                
                self.sv_x.append(x)
                self.sv_w_inner.append(w)
            
                # Stopping criterion for inner cycle        
                if self.stop_criterion_inner(A, w, w_obs, x, xold):
                    break

            self.sv_x_out.append(x)
            
            # Stopping criterion for outer iterations
            if self.stop_criterion_outer(self.sv_w, self.sv_x_out, self.epsilon_out, i_out):
                break

        return x

