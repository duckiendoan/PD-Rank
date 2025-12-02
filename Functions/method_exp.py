import numpy as np
from scipy.optimize import newton
from scipy.special import softmax, expit

def cost_func(A, w, x, gamma):
    return np.dot(w, np.logaddexp(0, 1 - A @ x)) + gamma * np.linalg.norm(x) ** 2

def relative_norm_change(x, x_old):
    return np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
                                                                          
def get_tau_sigma(A):
    tau = 1 / (2 * A.shape[0]) ** 0.5
    tau_PDGH, sigma_PDGH = tau, tau
    return tau_PDGH, sigma_PDGH

def prox_tau_f(u, reg_term):
    u = u * reg_term
    return (u - np.mean(u)).ravel()

def func_h(u, x, w):
    # Equation 4.14 in the paper
    return -np.exp(1 - u) / (1 + np.exp(1 - u)) + (u - x) / w

def dh_du(u, x, w):
    return np.exp(1 - u) / (1 + np.exp(1 - u)) ** 2 + 1 / w

def prox_w_g(x, w):
    # Find a good initial value x0 for equation 4.14
    x_init = x.copy()
    temp = x + w
    temp2 = 2*(3*x+2*w)/(6+w)
    cond1 = (x < 4) & (temp < -1)
    cond2 = (x < 4) & (temp >= -1)
    x_init[cond1] = temp[cond1]
    x_init[cond2] = temp2[cond2]
    sols = newton(func_h, x_init, fprime=dh_du, args=(x, w), maxiter=50)
    return sols

def prox_sigma_g(x, sigma, tau, w):
    x_tilded = x / sigma
    w_tilded = w / sigma
    prox_w_tilded_g = prox_w_g(x_tilded, w_tilded)
    return x - tau * prox_w_tilded_g

class PDRankSolver:
    def __init__(self, rho = 1.9, step_ratio = 1, reg_gamma = 1e-4,
                    epsilon = 1e-2, inner_max_iter = 500, outer_max_iter = 25,
                    epsilon_in = 1e-2, epsilon_out = 1e-2):
        self.rho = rho
        self.step_ratio = step_ratio
        self.reg_gamma = reg_gamma
        self.epsilon = epsilon
        self.inner_max_iter = int(inner_max_iter)
        self.outer_max_iter = int(outer_max_iter)
        self.epsilon_in = epsilon_in
        self.epsilon_out = epsilon_out

    def solve(self, A, w_obs = None, return_solution_path=False):
        n, m = A.shape
        # Init
        x_init = np.random.randn(m)
        x_init = softmax(x_init)
        w = np.ones(n)
        w_obs = w_obs if w_obs is not None else np.ones(n)

        # Parameters
        tau_PDGH, sigma_PDGH = get_tau_sigma(A)
        step_size = self.rho
        reg_term = 1 / (1 + 2 * self.reg_gamma) # regularization term

        # Solution
        x = x_init.copy()
        z = A.dot(x)

        # Storage
        ws = np.zeros(shape=(self.outer_max_iter, *w.shape))
        xs = np.zeros(shape=(self.outer_max_iter, *x.shape))
        ws[0] = w

        for k in range(self.outer_max_iter):
            # Outer iteration: update w
            if k > 0:
                w = 1 / (np.logaddexp(0, 1 - A @ x) + self.epsilon)
            ws[k] = w.copy()
            # Inner iteration: solve for x = argmin ...
            for _ in range(self.inner_max_iter):
                # Update according to PDGH algorithm
                x_old = x
                # z_old = z.copy()
                x_bar = prox_tau_f(x - tau_PDGH * (A.T).dot(z), reg_term)
                z_bar = prox_sigma_g(z + sigma_PDGH * A.dot((2 * x_bar - x)), sigma_PDGH, tau_PDGH, w * w_obs)
                x = x + step_size * (x_bar - x)
                z = z + step_size * (z_bar - z)
                
                # Stop criterion:
                cost_new = cost_func(A, w * w_obs, x, self.reg_gamma)
                cost_old = cost_func(A, w * w_obs, x_old, self.reg_gamma)
                relative_change =  np.abs(cost_new - cost_old) / cost_old
                if relative_change < self.epsilon_in or cost_new < 1e-9:
                    break

            xs[k] = x.copy()
            # Stop criterion for outer iteration
            if k > 1:
                delta_w = relative_norm_change(ws[k], ws[k - 1])
                delta_x = relative_norm_change(xs[k], xs[k - 1])
                if delta_w < self.epsilon_out or delta_x < self.epsilon_out:
                    break

        if return_solution_path:
            return x, ws[:k+1], xs[:k+1]

        return x