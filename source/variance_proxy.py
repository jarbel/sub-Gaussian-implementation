from scipy.optimize import root_scalar, minimize_scalar
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import warnings


class AdaptiveSearchWarning(UserWarning):
    """Indicates the adaptive optimization failed to converge properly."""
    pass

# P(X>t) < exp(-t²/2σ²)

def subgaussian_proxy_variance_bernoulli(p: float) -> float:
    """
    Compute the optimal sub-Gaussian  variance proxy for a Bernoulli(p) distribution.
    
    Parameters:
    ----------
    - p: float, probability of success (0 < p < 1)

    Returns:
    - float: The optimal variance proxy sigma_opt_squared
    """
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1 (exclusive).")
    
    if np.isclose(p, 0.5, rtol=0, atol=1e-12):
        sigma_opt_squared =  1/4
    else:
        sigma_opt_squared = (0.5 - p) / np.log(1.0 / p - 1)

    return sigma_opt_squared



def subgaussian_proxy_variance_binomial(n: int, p: float) -> float:
    """
    Compute the optimal sub-Gaussian variance proxy  for a Binomial(n,p) distribution.
    For S = sum_{i=1}^n X_i with X_i i.i.d. Bernoulli(p),
    the optimal variances proxy  add: sigma_opt^2(S) = n * sigma_opt^2(Bernoulli(p)).
    
    Parameters:
    ----------
    - p: float, probability of success (0 < p < 1)

    Returns:
    - float: optimal variance proxy sigma_opt_squared
    """

    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1 (exclusive).")
    
    if n==0:
        return 0
    
    return n * subgaussian_proxy_variance_bernoulli(p)


def subgaussian_proxy_variance_uniform(a: float, b: float) -> float:
    
    """
    Compute the optimal sub-Gaussian variance proxy  for Uniform(a, b).

    X ~ Uniform(a, b) has optimal proxy:
        sigma_opt² = Var(X) = (b - a)² / 12.

    Parameters:
    ----------
        a: Lower bound of the interval.
        b: Upper bound of the interval (must satisfy b > a).

    Returns:
        float: The optimal sub-Gaussian variance proxy  (equals the variance).
    """

    if a >= b:
        raise ValueError("b must be greater than a" )

    return 1/12 * (b-a)**2

def subgaussian_proxy_variance_sum_independant_uniform(segments: tuple) -> float:
    
    """
    Parameters:
    ----------
    segments: list of tuples [(a1, b1), (a2, b2), ...] representing independent Uniform(a, b) 
              but not necessarily identically distributed
              
    Returns: 
    total mean, variance, and sub-Gaussian variance proxy of the sum
    """
    if not segments or not all(isinstance(seg, tuple) and len(seg) == 2 for seg in segments):
        raise ValueError("segments must be a non-empty list of tuples (a, b).")
    
    total_sigma2_opt = 0

    for i, (a, b) in enumerate(segments):
        if a >= b :
            raise ValueError(f"Invalid interval at position {i}: a={a}, b={b}") 
    
        sigma_opt_squared_i = subgaussian_proxy_variance_uniform(a, b)
        total_sigma2_opt += sigma_opt_squared_i
    
    return total_sigma2_opt

def subgaussian_discrete_uniforme_variance_proxy(a: float, n: int) -> float:
    """
    Variance proxy  for a discrete uniform with equally spaced support:

        X ∈ {h + a*k : k = 0, 1, ..., n-1}

    The variance is independent of the offset h and equals:

        Var[X] = a^2 * (n^2 - 1) / 12

    Parameters
    ----------
    a : float
        Spacing between support points.
    n : int
        Number of support points (must be >= 2).
    
    Returns:
        - float: optimal sub-Gaussian variance proxy

    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    
    return (a**2 * (n**2 - 1.0)) / 12.0
    

def subgaussian_proxy_variance_truncated_random(a: float, b: float, mu: float, sigma_opt_squared: float) -> float:
    """
    Compute the optimal sub-Gaussian variance proxy for a truncated normal variable.

    Parameters:
    - a, b: float, bounds of truncation interval (a < b)
    - mu: float, mean of the original normal variable
    - sigma2: float, variance of the original normal variable (σ² > 0)

    Returns:
    - float: optimal sub-Gaussian variance proxy
    """
    if  a < b:
        raise ValueError("Invalid interval: require a < b")
    sigma = np.sqrt(sigma_opt_squared)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    if a + b == 2 * mu:
        z = beta
        numerator = norm.pdf(z)
        denominator = 2 * norm.cdf(z) - 1
        proxy_variance = sigma_opt_squared * (1 - 2 * z * numerator / denominator)
    else:
        factor = 2 * sigma_opt_squared / (b + a - 2 * mu)
        numerator = norm.pdf(alpha) - norm.pdf(beta)
        denominator = norm.cdf(beta) - norm.cdf(alpha)
        proxy_variance = sigma_opt_squared * (1 - factor * numerator / denominator)
    return proxy_variance



class SubGaussianTriangularProxy:
    """
    Triangular distribution with: 
    a, b: positive parameters defining the support (-a, b)
    Using proposition 2.4 in Arbel et al paper "On strict sub-Gaussianity, optimal proxy variance
                                                and symmetry for bounded random variables" 
    """
    
    def __init__(self, a, b):

        if a <= 0 or b <= 0:
            raise ValueError("Parameters a and b must be positive")
        
        self.a = a
        self.b = b
        self.mean = (b - a) / 3  
        self.variance = (a**2 + a*b + b**2) / 18 
        self.lower = self.variance
        self.hoeffding_bound = (self.a + self.b)**2 / 4 # (b-a)²/4 in (a,b) interval

        self.threshold_lambda_taylor = 1e-5
        self.threshold_lambda_mgf_1 = 1e-8
        self.tolerance = 1e-9
        self.max_iter = 100

    def _mgf_centred(self, lam):
        """
        Compute E[exp(λ(X - µ))] for the triangular distribution
        Uses numerically stable computation to avoid overflow
        """
        # Use Taylor series for small λ for numerical stability
        if abs(lam) < self.threshold_lambda_taylor:
            return self._mgf_centered_series(lam)
                
        a, b = self.a, self.b
        
        if abs(lam) < self.threshold_lambda_mgf_1:
            return 1.0
        
        # Compute the exponents
        exp1 = lam * (a + 2*b) / 3
        exp2 = -lam * (b + 2*a) / 3  
        exp3 = lam * (a - b) / 3
        
        max_exp = max(exp1, exp2, exp3)
        term1_stable = a * np.exp(exp1 - max_exp)
        term2_stable = b * np.exp(exp2 - max_exp)
        term3_stable = (a + b) * np.exp(exp3 - max_exp)
        
        numerator_stable = (term1_stable + term2_stable - term3_stable) * np.exp(max_exp)
        denominator = a * b * (a + b) * lam**2 / 2
        mgf = numerator_stable / denominator
        
        return mgf    


    def _mgf_centered_series(self, lam: float) -> float:
        """
        Taylor series expansion of MGF around λ = 0 for numerical stability
        E[exp(λ(X - μ))] = 1 + µ2*λ²/2 + μ3*λ^3/6 + μ4*λ^4/24 + ...
        where µk are the k-th central moments
        """
        mu2 = self.variance
        mu3 = (self.b - self.a) * (2*self.a + self.b) * (2*self.b + self.a) / 270.0
        mu4 = (self.a**2 + self.a*self.b + self.b**2)**2 / 135.0
        l2 = lam * lam
        return 1.0 + 0.5 * mu2 * l2 + (mu3 * lam * l2) / 6.0 + (mu4 * l2 * l2) / 24.0
    
    def _mgf_derivative(self, lam: float) -> float:
        if abs(lam) < self.threshold_lambda_taylor:
            mu2 = self.variance
            mu3 = (self.b - self.a) * (2*self.a + self.b) * (2*self.b + self.a) / 270.0
            mu4 = (self.a**2 + self.a*self.b + self.b**2)**2 / 135.0
            l2 = lam * lam
            return mu2 * lam + 0.5 * mu3 * l2 + (mu4 / 6.0) * lam * l2
        
        a, b = self.a, self.b
        
        # Derivative of the complex MGF expression
        # This is the derivative of: 2/(ab(a+b)λ²) * [aexp(λ(a+2b)/3} + bexp(-λ(b+2a)/3} - (a+b)exp(λ(a-b)/3}]

        exp1 = np.exp(lam * (a + 2*b) / 3)
        exp2 = np.exp(-lam * (b + 2*a) / 3)
        exp3 = np.exp(lam * (a - b) / 3)
        
        bracket_term = a * exp1 + b * exp2 - (a + b) * exp3
        bracket_derivative = (a * (a + 2*b) / 3) * exp1 + (b * (-(b + 2*a)) / 3) * exp2 - ((a + b) * (a - b) / 3) * exp3
        
        denominator = a * b * (a + b) * lam**2
        numerator = 2 * bracket_derivative * denominator - 2 * bracket_term * 2 * a * b * (a + b) * lam
        result = numerator / (denominator**2)
        
        return result


    
    def delta_function(self, sigma2, lam):
        """
        Compute the Δ function from Proposition 2.4:
        Δ(σ², λ) = exp(λ²σ²/2) - E[exp(λ(X - μ))]
        """
        gaussian_term = np.exp(lam**2 * sigma2 / 2)
        mgf = self._mgf_centred(lam)
        return gaussian_term - mgf

    def delta_derivative(self, sigma2, lam):
        """
        Compute dΔ/dλ for finding critical points
        """

        gaussian_term_derivative = lam * sigma2 * np.exp(lam**2 * sigma2 / 2)
        
        mgf_derivative = self._mgf_derivative(lam)
        
        return gaussian_term_derivative - mgf_derivative

    
    def check_delta_conditions(self, sigma2, lam):
        """
        Check both conditions from Proposition 2.4:
        1. Δ(σ², λ) = 0
        2. d_λΔ(σ², λ) = 0
        """
        delta_val = self.delta_function(sigma2, lam)
        delta_deriv = self.delta_derivative(sigma2, lam)
        return delta_val, delta_deriv

    def _min_delta_over_lambda(self, sigma2, L0=4.0, L_max=1e4, step=0.02, edge_margin_pts=5):
        """
        Find min_lambda Δ(σ², λ) with adaptive expansion of [-L, L].
        Start from L0 and double while the minimum sticks to the boundary.
        Returns (min_delta, argmin_lambda, L_used).
        """
        L = float(L0)
        best = None  # (val, lam, L)
        while L <= L_max:
            n_pts = int(2 * L / step) + 1
            grid = np.linspace(-L, L, max(n_pts, 201))  # au moins 201 points
            vals = np.array([self.delta_function(sigma2, lam) for lam in grid])
            k = int(np.argmin(vals))
            lam_star0 = grid[k]

            if k <= edge_margin_pts or k >= len(grid) - 1 - edge_margin_pts:
                best = (vals[k], lam_star0, L) if (best is None or vals[k] < best[0]) else best
                L *= 2.0
                continue

            left = grid[max(0, k - 5)]
            right = grid[min(len(grid) - 1, k + 5)]
            res = minimize_scalar(lambda lam: self.delta_function(sigma2, lam),
                                bounds=(left, right), method='bounded')
            if res.success:
                return res.fun, res.x, L

            return vals[k], lam_star0, L

        if best is not None:
            return best
        return vals[k], lam_star0, L_max


    def subgaussian_variance_proxy(self, debug=False):
        """ Smallest σ² in [Var[X], Hoeffding] such that min_λ Δ(σ²,λ) >= 0. 
        """
        if abs(self.a - self.b) < self.tolerance:
            self.sigma_opt_squared = self.variance
            return self.sigma_opt_squared
            
        lo = float(self.variance)
        hi = float(self.hoeffding_bound)

        val_hi, lam_hi, _ = self._min_delta_over_lambda(hi)
        if val_hi < -self.tolerance:
            warnings.warn(
                    "[sigma2_opt_beta] Optimizer stuck at boundary Consider refining step, or improving numerical stability.",
                    AdaptiveSearchWarning,
                    stacklevel=2
                )
            if debug:
                return {
                    'optimal_proxy_variance': None,
                    'variance': self.variance,
                    'delta_at_hoeffding': val_hi,
                    'min_feasible': False,
                    'reason': 'min_λ Δ(σ²,λ) < 0 à σ² = Hoeffding; verif Δ or L_max.'
                }
            return None
        

        best_lam = lam_hi
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            val_mid, lam_mid, _ = self._min_delta_over_lambda(mid)
            if val_mid >= -self.tolerance:
                hi = mid
                best_lam = lam_mid
            else:
                lo = mid

            if hi - lo <= self.tolerance * max(1.0, abs(hi)):
                break

        sigma_opt_squared = hi
        min_delta, lam0, _ = self._min_delta_over_lambda(sigma_opt_squared)


        if debug:
            return {
                'optimal_proxy_variance': sigma_opt_squared,
                'variance': self.variance,
                'delta_at_critical_point': min_delta,             
                'delta_derivative_at_critical_point': 0.0,   
                'min_delta_over_range': min_delta,
                'non_negative_condition_satisfied': (min_delta >= -self.tolerance),
                'critical_point_lambda': lam0,
                'is_strictly_subgaussian': abs(sigma_opt_squared - self.variance) < self.tolerance
            }

        return sigma_opt_squared



class SubGaussian3MassSymetricProxy:

    """
    Class for computing the optimal sub-Gaussian variance proxy for 
    symmetric 3-mass discrete distributions on {-a, 0, +a}
    with probabilities: p at -a, 1-2p at 0, p at +a.

    Attributes:
    ----------
    p : float
        The probability parameter (must satisfy 0 < p < 1).
    a : float
        The scaling parameter for the support points.

    Returns:
    -------
    sigma_opt_squared : float
        The computed optimal variance proxy (initialized after computation).
        
    """

    def __init__(self, p: float, a: float = 1):
        if not 0 < p <= 1/2:
            raise ValueError("p must be in (0,0.5].")
        self.p = p
        self.a = a
        self.sigma_opt_squared = None 
        self.lambda_star = None
        self.lambda_0 = np.arccosh(
            (1 - 4 * self.p  - 4 * self.p  ** 2) / (2 * self.p  * (1 - 2 * self.p )))  
        self.lower_bound = 2 * self.p 
        self.upper_bound = (1 - 2 * self.p ) ** 2  / (4 * (1 - 4 * self.p )) 

    def _equation(self, lambda_c):
        term = 2 * self.p * np.cosh(lambda_c) + 1 - 2 * self.p
        equation = self.p * lambda_c * np.sinh(lambda_c) - term * np.log(term)
        return equation

    def plot_objective_function(self):
        
        lambdas = np.linspace(self.lambda_star - 1, self.lambda_star + 1 , 5000)
        equations = [self._equation(lam) for lam in lambdas]

        plt.figure(figsize=(8, 5))
        plt.plot(lambdas, equations, label=f"p={self.p}, a=1")
        plt.axhline(0, color='gray', lw=0.5, ls='--')
        plt.title(
            r"$p \, \lambda_c \sinh(\lambda_c) - "
            r"(1 - 2p + 2p \cosh(\lambda_c)) \, \ln(1 - 2p + 2p \cosh(\lambda_c)) = 0$"
        )
        plt.xlabel("λ")
        plt.ylabel("Objective Value")
        plt.legend()
        plt.grid()
        plt.show()

    def subgaussian_variance_proxy(self, tol=1e-7):
        if self.p >= 1./6:
            self.sigma_opt_squared = self.lower_bound
    
        else:
            lambdas = np.linspace(self.lambda_0 + tol, 50, 5000)
            signs = np.sign([self._equation(lam) for lam in lambdas])

            for i in range(len(signs) - 1):
                if signs[i] != signs[i + 1]:
                    a, b = lambdas[i], lambdas[i + 1]
                    result = root_scalar(
                        self._equation, bracket=[a, b], method='bisect', xtol=tol
                        )
                    if result.converged:
                        self.lambda_star = result.root
                        denom = 2 * self.p * np.cosh(self.lambda_star) + 1 - 2 * self.p
                        self.sigma_opt_squared = (
                            2 * self.p * np.sinh(self.lambda_star)
                        ) / (self.lambda_star * denom)
                    else:
                        raise RuntimeError("Root-finding did not converge.")
                    break
            else:
                self.sigma_opt_squared  = np.nan
                self.lambda_star = np.nan
                warnings.warn(f"No sign change found; root cannot be located for p = {self.p}")

        return self.a**2 * self.sigma_opt_squared, self.lambda_star
    

class SubGaussian3MassAssymetricProxy:
    """
    Class for computing the optimal sub-Gaussian variance proxy for assymetric 3-mass distribution on {-a, 0, +a}
    with probabilities: p1 at -a, p3=1-p1-p2 at 0, p2 at +a.
    
    Attributes:
    ----------
    p1 : float
        The probability parameter corresponding to -a (must satisfy 0 < p1 < 1).
    p2 : float
        The probability parameter corresponding to +a (must satisfy 0 < p2 < 1).
        
    a : float
        The scaling parameter for the support points.
    
    Returns:
    -------
    sigma_opt_squared : float
        The computed optimal variance proxy (initialized after computation).
   
    """


    def __init__(self, p1: float, p2: float, a: float):
        if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
            raise ValueError("p1 and p2 must be in (0,1).")
        if p2 < p1:
            raise ValueError("p2 must be >= p1.")
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.p3 = 1.0 - self.p1 - self.p2
        if not (0.0 < self.p3 < 1.0):
            raise ValueError("p3 must be in (0,1), i.e., p1 + p2 < 1.")
        self.a = float(a)

        self.variance = self.p1 + self.p2 - (self.p2 - self.p1) ** 2
        self.sigma_opt_squared = None


    def _logu0_and_r(self, lam: float):
        """
        Return numerically stable (log_u0, r=u1/u0) for λ>0.
        Where:
            u0(λ) = p1 exp(-λ) + p2 exp(λ) + p3
            u1(λ) = -p1 exp(-λ) + p2 exp(λ)
            r     = u1/u0 = -w1 + w2 where w_i are stable weights in [0,1].
        """
        
        if lam <= 0.0:
            return np.nan, np.nan

        t1 = np.log(self.p1) - lam
        t2 = np.log(self.p2) + lam
        t3 = np.log(self.p3)
        m = max(t1, t2, t3)
        s = np.exp(t1 - m) + np.exp(t2 - m) + np.exp(t3 - m) 
        log_u0 = m + np.log(s)
        w1 = np.exp(t1 - log_u0)  # p1 e^{-λ} / u0
        w2 = np.exp(t2 - log_u0)  # p2 e^{λ} / u0
        r = -w1 + w2
        return log_u0, r


    def _equation(self, lam: float) -> float:
        """
        G(λ) = λ * (u1/u0) - 2 * log(u0) + λ * (p2 - p1).
        Note: F(λ) = u0 * G(λ) and u0>0,  solving G(λ)=0 is equivalent and stable.
        """
        log_u0, r = self._logu0_and_r(lam)
        if not np.isfinite(log_u0) or not np.isfinite(r):
            return np.nan
        return lam * r - 2.0 * log_u0 + lam * (self.p2 - self.p1)


    def _default_proxy_first_regime(self) -> float:
        """
        Closed-form value in the easy regime (boundary at λ → 0+).

        Returns
        -------
        float
            The proxy variance in the easy regime.
        """
        try:
            if self.p3 > 4.0 * np.sqrt(self.p1 * self.p2):
                raise ValueError(
                    "Not in the easy regime: requires p3 <= 4 * sqrt(p1 * p2)."
                )

            if np.isclose(self.p1, self.p2):
                return self.variance

            return 2.0 * (self.p2 - self.p1) / np.log(self.p2 / self.p1)

        except (ZeroDivisionError, FloatingPointError, ValueError) as e:
            return float("nan")

    def _lambda_minus(self):

        d1 = self.p3**2 - 4.0 * self.p1 * self.p2
        d2 = self.p3**2 - 16.0 * self.p1 * self.p2
        if d1 <= 0.0 or d2 <= 0.0:
            return None
        num = self.p3**2 - 8.0 * self.p1 * self.p2 - np.sqrt(d1 * d2)
        den = 2.0 * self.p1 * self.p2

        if den <= 0.0 or num <= 0.0:
            return None
        return float(np.log(num / den))

    def _bracket_root(self,
                      lam_min: float = 1e-12,
                      lam_max: float = 500.0,
                      growth: float = 1.8,
                      max_iter: int = 200):

        p_eff = max(self.p1, self.p2, 1e-15)
        lam_lo = float(np.clip(np.sqrt(1e-6 / p_eff), lam_min, 1.0))  # numerically useful low end
        lam_hi = lam_lo * 2.0
        lam_minus = self._lambda_minus()
        if lam_minus is not None:
            lam_hi = max(lam_hi, lam_minus)

        g_lo = self._equation(lam_lo)
        tries = 0
        while (not np.isfinite(g_lo)) and lam_lo > lam_min and tries < 20:
            lam_lo *= 0.5
            g_lo = self._equation(lam_lo)
            tries += 1

        g_hi = self._equation(lam_hi)

        it = 0
        while (np.isfinite(g_lo) and np.isfinite(g_hi)
               and np.sign(g_lo) == np.sign(g_hi)
               and lam_hi < lam_max and it < max_iter):
            lam_lo, g_lo = lam_hi, g_hi
            lam_hi = min(lam_hi * growth, lam_max)
            g_hi = self._equation(lam_hi)
            it += 1

        if (np.isfinite(g_lo) and np.isfinite(g_hi)
                and np.sign(g_lo) != np.sign(g_hi)):
            return (min(lam_lo, lam_hi), max(lam_lo, lam_hi))

        return None


    def subgaussian_variance_proxy(self, tol: float = 1e-8) -> float:
        """
        Return a^2 * sigma_opt_squared (no NaN):
        - Easy regime (p3 <= 4*sqrt(p1*p2)): exact closed-form (boundary at λ→0+).
        - Hard regime (p3  > 4*sqrt(p1*p2)):
            * If an interior root exists: solve G(λ)=0 (Brent).
            * If no interior root on (0, λ_max]: optimum lies at a boundary:
                · if G(λ)>0 throughout → maximum at upper boundary,
                · if G(λ)<0 throughout → maximum at λ→0^+ (closed-form).
        """
     

        if self.p3 <= 4.0 * np.sqrt(self.p1 * self.p2):
            self.sigma_opt_squared = self._default_proxy_first_regime()
            return self.a**2 * self.sigma_opt_squared


        bracket = self._bracket_root()
        if bracket is not None:
            sol = root_scalar(self._equation, bracket=bracket, method='brentq', xtol=tol)
            if not (sol.converged and sol.root > 0.0):
                raise RuntimeError("Brent failed to converge on a valid bracket for G(λ)=0.")
            lam = float(sol.root)
            _, r = self._logu0_and_r(lam)
            if np.isclose(self.p1, self.p2, atol=1e-12, rtol=0.0):
                self.sigma_opt_squared = r / lam
            else:
                self.sigma_opt_squared = max(2.0 * (self.p2 - self.p1) / np.log(self.p2 / self.p1) , (r - (self.p2 - self.p1)) / lam)
            return self.a**2 * self.sigma_opt_squared


        # lam_upper = self._lambda_minus() or 700.0
        # # Evaluate G at both ends (numerically meaningful near 0)
        # lam_low_eval = max(1e-12, np.sqrt(1e-12 / max(self.p1, self.p2, 1e-15)))
        # g_low = self._equation(lam_low_eval)
        # g_up = self._equation(lam_upper)

        # if np.isfinite(g_up) and g_up > 0.0 and (not np.isfinite(g_low) or g_low >= 0.0):
        #     # Monotone positive → optimum at upper boundary
        #     _, r_up = self._logu0_and_r(lam_upper)
        #     if np.isclose(self.p1, self.p2, atol=1e-12, rtol=0.0):
        #         self.sigma_opt_squared = r_up / lam_upper
        #     else:
        #         self.sigma_opt_squared = (r_up - (self.p2 - self.p1)) / lam_upper
        # else:
        #     self.sigma_opt_squared = self.variance if self.p1 == self.p2 else self._default_proxy_first_regime()

        # return self.a ** 2 * self.sigma_opt_squared



class SubGaussianBetaProxy:
    def __init__(self, alpha, beta):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Parameters must be positive")
        self.alpha = alpha
        self.beta = beta 
        self.var = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        self.mu = self.alpha / (self.alpha + self.beta)

        self.bounds_list = [2, 5, 10, 20, 50, 100]  
        self.bracket_scales = [1, 2, 5, 10, 20, 40]  

    def h_beta(self, lam):
        """Compute h(λ) with safe fallback."""
        if abs(lam) < 1e-14:
            return self.var
        
        try:
            val = np.exp(-lam * self.mu) * hyp1f1(self.alpha, self.alpha + self.beta, lam)
            if val > 0 and np.isfinite(val):
                result = 2.0 / (lam * lam) * np.log(val)
                if np.isfinite(result):
                    return result
        except Exception:
            pass
        
        return -np.inf  # Safe fallback for any error

    def plot_objective_function(self, lam_min=-100, lam_max=100, n_points=50000):
            """
            Plot h(λ) = 2/λ² * log E[exp(λ(X-μ))] and its maximum.
            """
            
            lam_vals = np.linspace(lam_min, lam_max, n_points)
            h_vals = [self.h_beta(l) for l in lam_vals]

            opt_val, lam_star = self.subgaussian_variance_proxy()

            plt.figure(figsize=(7, 4))
            plt.plot(lam_vals, h_vals, label="h(λ)")
            plt.axvline(lam_star, color="gray", ls="--", label=f"λ*={lam_star:.2f}")
            if np.isfinite(lam_star):
                plt.scatter([lam_star], [opt_val], color="red", zorder=5,
                            label=f"$\max_{{\lambda}} h={opt_val:.4f} \ at \ \lambda^*={lam_star:.2f}$")
            plt.xlabel("λ")
            plt.ylabel("h(λ)")
            plt.title(f"h(λ) for Beta(α={self.alpha}, β={self.beta})")
            plt.legend(loc="lower left")
            plt.grid(True)
            plt.show()    
            
    def subgaussian_variance_proxy(self):
        """
        Adaptive bound search for σ²_opt = max_λ h(λ) where
        h(λ) = 2/λ² * log( E[exp(λ(X-μ))] ) with X ~ Beta(alpha, beta).

        Returns
        -------
        sigma_opt_squared : float
            Optimale value of  h(λ) (variance proxy).
        lambda_star : float
            λ maximising h(λ).
        """
        # Symmetric case  : optimum at λ = 0
        if np.isclose(self.alpha, self.beta):
            lambda_star = 0.0
            sigma_opt_squared = self.var
            return sigma_opt_squared, lambda_star

        # Default initialization
        sigma_opt_squared = self.var
        lambda_star = 0.0

        # Search with Brent in brackets
        for scale in self.bracket_scales:
            try:
                bracket = (-scale, 0, scale)
                result = minimize_scalar(
                    lambda lam: -self.h_beta(lam),
                    bracket=bracket,
                    method='brent'
                )
                if result.success and np.isfinite(result.fun):
                    optimal_value = -result.fun
                    lambda_star = float(result.x)
                    if optimal_value >= self.var * 0.999:
                        return optimal_value, lambda_star
            except Exception:
                continue

        # Fallback search with Bounded Optimization
        for bound in self.bounds_list:
            try:
                result = minimize_scalar(
                    lambda lam: -self.h_beta(lam),
                    bounds=(-bound, bound),
                    method='bounded'
                )
                if result.success and np.isfinite(result.fun):
                    optimal_value = -result.fun
                    if optimal_value >= self.var * 0.999 and optimal_value > sigma_opt_squared:
                        sigma_opt_squared = optimal_value
                        lambda_star = float(result.x)
                        if abs(result.x) < 0.9 * bound:
                            break
            except Exception:
                continue  

        return sigma_opt_squared, lambda_star
