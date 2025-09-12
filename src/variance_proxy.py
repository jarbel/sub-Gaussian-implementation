from scipy.optimize import root_scalar, minimize_scalar, brentq
from scipy.special import hyp1f1, betaln, gamma as gammaln
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import warnings
import math


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
    the optimal variance proxies add: sigma_opt^2(S) = n * sigma_opt^2(Bernoulli(p)).
    
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
        raise ValueError("b must be greater than a")

    return 1/12 * (b-a)**2

def subgaussian_proxy_variance_sum_independent_uniform(segments: tuple) -> float:
    
    """
    Parameters:
    ----------
    segments: list of tuples [(a1, b1), (a2, b2), ...] representing independent Uniform(a, b) 
              but not necessarily identically distributed
              
    Returns: 
    sub-Gaussian variance proxy of the sum
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

def subgaussian_discrete_uniform_variance_proxy(a: float, n: int) -> float:
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
    

def subgaussian_proxy_variance_truncated_normal(a: float, b: float, mu: float, sigma_opt_squared: float) -> float:
    """
    Compute the optimal sub-Gaussian variance proxy for a truncated normal variable.

    Parameters:
    - a, b: float, bounds of truncation interval (a < b)
    - mu: float, mean of the original normal variable
    - sigma2: float, variance of the original normal variable (σ² > 0)

    Returns:
    - float: optimal sub-Gaussian variance proxy
    """
    if  a >= b:
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

def subgaussian_proxy_variance_truncated_exponential(a: float, b: float, lam: float) -> float:
    """
    Compute the optimal sub-Gaussian variance proxy for a truncated exponential variable.
    
    Parameters:
    - a: float, lower bound of truncation interval (a ≥ 0)
    - b: float, upper bound of truncation interval (b > a)  
    - lam: float, rate parameter of exponential distribution (λ > 0)
    
    Returns:
    - float: optimal sub-Gaussian variance proxy
    
    ----------
    """

    if lam <= 0:
        raise ValueError("Rate parameter λ must be positive")
    if a < 0:
        raise ValueError("Lower bound a must be non-negative")
    if a >= b:
        raise ValueError("Invalid interval: require 0 ≤ a < b")
    if b == np.inf:
        raise ValueError("Upper bound b must be finite for truncated exponential")
    
    exp_lambda_a = np.exp(lam * a)
    exp_lambda_b = np.exp(lam * b)
    
    if exp_lambda_b - exp_lambda_a == 0:
        raise ValueError("Numerical issue: exp(λb) - exp(λa) is zero")
    
    numerator = (b - a) * (exp_lambda_b + exp_lambda_a)
    denominator = 2 * lam * (exp_lambda_b - exp_lambda_a)
    
    proxy_variance = numerator / denominator - 1 / (lam**2)
    
    return proxy_variance



@dataclass
class SubGaussianTriangularProxy:
    """
    Class to compute the optimal sub-Gaussian variance proxy
    for a triangular distribution on (-a, b).
    The search is performed on a grid of σ² and, for each σ²,
    on a grid of λ to find the roots of Δ(σ², λ) = 0.
    The pair (σ², λ) that minimizes |Δ′(σ², λ)| is retained.

    Attributes:
    ----------
    - a, b: bounds of the triangular distribution (a > 0, b > 0)
    
    - interval_search_bound: absolute bound for λ and σ² search
    - n_lambda_grid: number of points to discretize λ
    - enforce_nonneg_delta: enforce Δ ≥ 0 over the entire λ interval
    - n_sigma_refine : Grid points for σ² refinement in each iteration.
    - lambda_max : Maximum |λ| value to consider (reduced from 100 for efficiency).
    - max_iter : Maximum iterations for root finding (reduced from 200).
    - lambda_threshold_taylor : Threshold for Taylor series vs. closed form MGF calculation.
    - precision : numerical tolerance for root tests and underflow protection
    
    Returns:
    -------
    candidate : Dict[str, Any]
        Information about the optimal variance proxy search:
        - sigma_opt_squared : float
            The computed optimal variance proxy (initialized after computation).
        - lambda_star : float
            The critical point λ* where the optimal proxy is achieved

    """
 
    a: float
    b: float
    n_sigma_refine: int = 40
    lambda_max: float = 100.0
    max_iter: int = 200
    interval_search_bound: float = 200.0
    n_lambda_grid: int = 4001
    lambda_threshold_taylor: float = 1e-3
    lambda_search_bound: float = 1e-3
    precision: float = 1e-10
    max_passes: int = 4
    enforce_nonneg_delta: bool = True

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("Parameters a and b must be positive.")
        self.variance = (self.a**2 + self.a*self.b + self.b**2) / 18.0
        self.upper = ((self.a + self.b) ** 2) / 4.0
        # cap for exponent to avoid overflow
        self.lambda_exponent_cap = float(np.log(np.finfo(float).max))
        c1 = (self.a + 2 * self.b) / 3.0
        c2 = (self.b + 2 * self.a) / 3.0
        c3 = abs(self.a - self.b) / 3.0
        max_c = max(abs(c1), abs(c2), abs(c3))
        # Safe λ bound: ensure |λ|max * max_c <= log(max_float)
        if max_c > 0:
            lam_safe = self.lambda_exponent_cap / max_c
        else:
            lam_safe = self.lambda_exponent_cap
            
        self.interval_search_bound = min(self.interval_search_bound, lam_safe)
        self.lambda_max = min(self.lambda_max, lam_safe)

    # Internal helper functions
    def _exponential_term(self, sigma2: float, lam: float) -> float:
        x = 0.5 * lam * lam * sigma2
        if x > self.lambda_exponent_cap:
            return float('inf')
        if x < -self.lambda_exponent_cap:
            return 0.0
        return math.exp(x)

    def _central_moments(self) -> Tuple[float, float, float]:
        a, b = self.a, self.b
        mu2 = self.variance
        mu3 = (b - a) * (2 * a + b) * (2 * b + a) / 270.0
        mu4 = (a * a + a * b + b * b) ** 2 / 135.0
        return mu2, mu3, mu4

    def _mgf_centered_series(self, lam: float) -> Tuple[float, float]:
        mu2, mu3, mu4 = self._central_moments()
        l2 = lam * lam
        M = 1.0 + 0.5 * mu2 * l2 + (mu3 * lam * l2) / 6.0 + (mu4 * l2 * l2) / 24.0
        M1 = mu2 * lam + 0.5 * mu3 * l2 + (mu4 / 6.0) * lam * l2
        return M, M1

    def _mgf_centred_closed(self, lam: float) -> Tuple[float, float]:
        # coefficients c1, c2, c3 following the closed form derivation
        c1 = (self.a + 2 * self.b) / 3.0
        c2 = -(self.b + 2 * self.a) / 3.0
        c3 = (self.a - self.b) / 3.0
        e1 = lam * c1
        e2 = lam * c2
        e3 = lam * c3
        m = max(e1, e2, e3)
        # compute exponentials of shifted exponents; safe values in [0,1]
        E1 = math.exp(e1 - m)
        E2 = math.exp(e2 - m)
        E3 = math.exp(e3 - m)
        # base combination B and B1 multiplied by exp(m)
        denom = self.a * self.b * (self.a + self.b)
        lam2 = lam * lam
        if m > self.lambda_exponent_cap:
            return float('inf'), float('inf')
        B = (self.a * E1 + self.b * E2 - (self.a + self.b) * E3) * math.exp(m)
        B1 = (self.a * c1 * E1 + self.b * c2 * E2 - (self.a + self.b) * c3 * E3) * math.exp(m)
        M = (2.0 / denom) * (B / lam2)
        dM = (2.0 / denom) * (B1 / lam2 - 2.0 * B / (lam2 * lam))
        return M, dM

    def _mgf_and_derivative(self, lam: float) -> Tuple[float, float]:
        if abs(lam) < self.lambda_threshold_taylor:
            return self._mgf_centered_series(lam)
        return self._mgf_centred_closed(lam)

    def _delta(self, sigma2: float, lam: float) -> float:
        mgf, _ = self._mgf_and_derivative(lam)
        exp_term = self._exponential_term(sigma2, lam)
        return exp_term - mgf

    def _delta_prime(self, sigma2: float, lam: float) -> float:
        _, dM = self._mgf_and_derivative(lam)
        dE = lam * sigma2 * self._exponential_term(sigma2, lam)
        return dE - dM

    def _roots_for_sigma(self, sigma2: float) -> List[float]:
        for _ in range(self.max_passes):
            L_scan = min(self.lambda_search_bound, self.lambda_max)
            xs = np.linspace(-L_scan, L_scan, self.n_lambda_grid, dtype=float)
            vals = np.array([self._delta(sigma2, float(x)) for x in xs], dtype=float)
            roots: set[float] = set()
            for i in range(len(xs) - 1):
                y1, y2 = vals[i], vals[i + 1]
                if not math.isfinite(y1) or not math.isfinite(y2):
                    continue
                if abs(y1) < self.precision:
                    roots.add(xs[i])
                    continue
                if np.sign(y1) == np.sign(y2):
                    continue
                a, b = xs[i], xs[i + 1]
                try:
                    r = brentq(lambda t: self._delta(sigma2, t), a, b, xtol=self.precision, maxiter=self.max_iter)
                    roots.add(r)
                except ValueError:
                    continue
            if roots:
                return sorted(roots)
            new_L = min(self.lambda_search_bound * 2, self.lambda_max)
            if new_L <= self.lambda_search_bound * (1.0 + self.precision):
                break
            self.lambda_search_bound = new_L
        return []

    def _search_optimal_sigma2_lambda_on_grid(self, sigmas: np.ndarray) -> Dict[str, Any]:
        best = {"sigma2": None, "lambda": None, "abs_dprime": float("inf"), "min_delta": None}
        for sigma_opt_squared in sigmas:
            roots = self._roots_for_sigma(sigma_opt_squared)
            if not roots:
                continue
            if self.enforce_nonneg_delta:
                xs = np.linspace(-self.interval_search_bound, self.interval_search_bound, self.n_lambda_grid, dtype=float)
                vals = np.array([self._delta(sigma_opt_squared, x) for x in xs], dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                if finite_vals.size and finite_vals.min() < -self.precision:
                    continue
            for r in roots:
                d1 = abs(self._delta_prime(sigma_opt_squared, r))
                if d1 < best["abs_dprime"]:
                    best.update({"sigma_opt_squared": sigma_opt_squared, "lambda": r, "abs_dprime": d1})
        return best

    def subgaussian_optimal_variance_proxy(self) -> Dict[str, Any]:
        if abs(self.a - self.b) <= self.precision:
            return {
                "sigma_opt_squared": self.variance,
                "lambda_opt": 0.0,
                "ok": True,
                "checks": {"delta": 0.0, "delta_prime": 0.0},
                "note": "symmetric distribution → σ²=V[X], λ=0",
            }
        low, high = self.variance, self.upper
        sigmas = np.linspace(low, high, int(self.interval_search_bound))
        candidate = self._search_optimal_sigma2_lambda_on_grid(sigmas)
        if candidate.get("sigma_opt_squared") is None:
            return {"ok": False, "reason": "no root of Δ found on coarse grid"}
        for _ in range(self.max_passes):
            sigma_opt_squared, lam = float(candidate["sigma_opt_squared"]), float(candidate["lambda"])
            dval = abs(self._delta(sigma_opt_squared, lam))
            dpr = abs(self._delta_prime(sigma_opt_squared, lam))
            if dval <= self.precision and dpr <= self.precision:
                return {
                    "sigma_opt_squared": sigma_opt_squared,
                    "lambda_opt": lam,
                    "ok": True,
                    "checks": {"delta": dval, "delta_prime": dpr},
                }
            span = max((high - low) / self.n_sigma_refine, 1e-6 * max(1.0, high))
            s_low = max(low, sigma_opt_squared - 2.0 * span)
            s_high = min(high, sigma_opt_squared + 2.0 * span)
            sigmas_ref = np.linspace(s_low, s_high, self.n_sigma_refine)
            candidate = self._search_optimal_sigma2_lambda_on_grid(sigmas_ref)
            if candidate.get("sigma_opt_squared") is None:
                break
        sigma_opt_squared, lam = float(candidate.get("sigma_opt_squared")), float(candidate.get("lambda"))
        return {
            "sigma_opt_squared": sigma_opt_squared,
            "lambda_opt": lam,
            "ok": False,
            "checks": {
                "delta": abs(self._delta(sigma_opt_squared, lam)),
                "delta_prime": abs(self._delta_prime(sigma_opt_squared, lam)),
            },
        }

    def plot_objective_function(self, sigma2=None, lam=None, window=2.0, n_points=500) -> None:
        if sigma2 is None or lam is None:
            result = self.subgaussian_optimal_variance_proxy()
            sigma2 = result.get("sigma_opt_squared")
            lam = result.get("lambda_opt")
            if sigma2 is None or lam is None:
                raise ValueError("Could not determine optimal sigma2 and lambda for plotting.")
        lam_range = np.linspace(lam - window, lam + window, n_points)
        delta_vals = np.array([self._delta(sigma2, l) for l in lam_range])
        delta_prime_vals = np.array([self._delta_prime(sigma2, l) for l in lam_range])
        plt.figure(figsize=(10, 6))
        plt.plot(lam_range, delta_vals, label=r"$\Delta(\sigma^2, \lambda)$", color="blue")
        plt.plot(lam_range, delta_prime_vals, label=r"$d\Delta/d\lambda$", color="orange")
        plt.axvline(lam, color="red", linestyle="--", label=rf"$\lambda^* = {lam:.4f}$")
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.title(rf"Delta and its derivative at $\sigma^2={sigma2:.6f}$")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"Value")
        plt.legend()
        plt.grid(True)
        plt.show()


class SubGaussian3MassSymmetricProxy:

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
    lambda_star : float
        The critical point λ* where the optimal proxy is achieved 
        
    """

    def __init__(self, p: float, a: float = 1):
        if not 0 < p < 1/2:
            raise ValueError("p must be in (0,0.5) (exclusive)")
        self.p = p
        self.a = a
        self.sigma_opt_squared = None 
        self.lambda_star = None
        self.lambda_0 = np.arccosh(
            (1 - 4 * self.p  - 4 * self.p  ** 2) / (2 * self.p  * (1 - 2 * self.p )))  
        self.lower_bound = 2 * self.p 
        self.variance = 2 * self.p * self.a**2
        self.upper_bound = (1 - 2 * self.p ) ** 2  / (4 * (1 - 4 * self.p )) 

    def _equation(self, lambda_c):
        term = 2 * self.p * np.cosh(lambda_c) + 1 - 2 * self.p
        equation = self.p * lambda_c * np.sinh(lambda_c) - term * np.log(term)
        return equation



    def subgaussian_optimal_variance_proxy(self, tol=1e-7):
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
                        warnings.warn("Root-finding did not converge in SubGaussian3MassSymmetricProxy.", UserWarning)
                        raise RuntimeError("Root-finding did not converge.")
                    break
            else:
                warnings.warn(f"No sign change found; root cannot be located for p = {self.p}", UserWarning)
                self.sigma_opt_squared  = np.nan
                self.lambda_star = np.nan

        return self.a**2 * self.sigma_opt_squared, self.lambda_star
    
    def plot_objective_function(self):
        if self.p >= 1./6:
            print("There is a closed form solution for p >= 1/6, no need to plot.")
            return
        
        if self.lambda_star is None or not np.isfinite(self.lambda_star):
            print("Cannot plot: lambda_star is not computed or invalid.")
            return
            
        lambdas = np.linspace(self.lambda_star - 1, self.lambda_star + 1, self.DEFAULT_N_POINTS)
        equations = [self._equation(lam) for lam in lambdas]

        plt.figure(figsize=(8, 5))
        plt.plot(lambdas, equations, label=f"p = ({self.p}), a = 1")
        plt.axhline(0, color='gray', lw=0.5, ls='--')

        if np.isfinite(self.lambda_star):
            plt.axvline(self.lambda_star, color="red", ls="--", lw=0.8)
        plt.plot(self.lambda_star, 0, 'ro', label=fr"$\lambda_c^* = {self.lambda_star:.4f}$")

        plt.title(
            r"$p \, \lambda_c \sinh(\lambda_c) - (1 - 2p + 2p \cosh(\lambda_c))$" + "\n" +
            r"$\times \ln(1 - 2p + 2p \cosh(\lambda_c)) = 0$"
        )
        plt.xlabel(r"$\lambda_c$")
        plt.ylabel(r"$F(\lambda_c)$")
        plt.legend(loc="lower left")
        plt.grid()
        plt.show()

class SubGaussian3MassAsymmetricProxy:
    """
    Class for computing the optimal sub-Gaussian variance proxy for asymmetric 3-mass distribution on {-a, 0, +a}
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
    lambda_star : float
        The critical point λ* where the optimal proxy is achieved.
   
    """


    def __init__(self, p1: float, p2: float, a: float):
        if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
            raise ValueError("p1 and p2 must be in (0,1).")
        if p2 < p1:
            raise ValueError("p2 must be >= p1.")
        self.p1 = p1
        self.p2 = p2
        self.p3 = 1.0 - self.p1 - self.p2
        if not (0.0 < self.p3 < 1.0):
            raise ValueError("p3 must be in (0,1), i.e., p1 + p2 < 1.")
        self.a = a

        self.variance = self.p1 + self.p2 - (self.p2 - self.p1) ** 2
        self.sigma_opt_squared = None
        self.lambda_star = None


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


    def subgaussian_optimal_variance_proxy(self, tol: float = 1e-8) -> Tuple[float, float]:
        """
        Return a^2 * sigma_opt_squared :
        - Easy regime (p3 <= 4*sqrt(p1*p2)): exact closed-form (boundary at λ→0+).
        - Hard regime (p3  > 4*sqrt(p1*p2)):
            * If an interior root exists: solve G(λ)=0 (Brent).
            * If no interior root on (0, λ_max]: optimum lies at a boundary:
                · if G(λ)>0 throughout → maximum at upper boundary,
                · if G(λ)<0 throughout → maximum at λ→0^+ (closed-form).
        """
     

        if self.p3 <= 4.0 * np.sqrt(self.p1 * self.p2):
            self.sigma_opt_squared = self._default_proxy_first_regime()
            return self.a**2 * self.sigma_opt_squared, 0


        bracket = self._bracket_root()

        if bracket is not None:
            sol = root_scalar(self._equation, bracket=bracket, method='brentq', xtol=tol)
            if not (sol.converged and sol.root > 0.0):
                warnings.warn("Brent failed to converge on a valid bracket for G(λ)=0 in SubGaussian3MassAsymmetricProxy.", UserWarning)
                raise RuntimeError("Brent failed to converge on a valid bracket for G(λ)=0.")
            self.lambda_star = float(sol.root)
            _, r = self._logu0_and_r(self.lambda_star)
            if np.isclose(self.p1, self.p2, atol=1e-12, rtol=0.0):
                self.sigma_opt_squared = r / self.lambda_star
            else:
                self.sigma_opt_squared = max(2.0 * (self.p2 - self.p1) / np.log(self.p2 / self.p1) , (r - (self.p2 - self.p1)) / self.lambda_star)
            return self.a**2 * self.sigma_opt_squared, self.lambda_star




    def plot_objective_function(self, n_points: int = 50000):
        if self.p3 <= 4.0 * np.sqrt(self.p1 * self.p2):
            print(f"There is a closed form for (p1, p2) = ({self.p1}, {self.p2}), no plot to display.")
            return  
        
        if self.lambda_star is None or not np.isfinite(self.lambda_star):
            print("Cannot plot: lambda_star is not computed or invalid.")
            return
            
        lambdas = np.linspace(self.lambda_star - 1, self.lambda_star + 1 , n_points)
        equations = [self._equation(lam) for lam in lambdas]

        plt.figure(figsize=(8, 5))
        plt.plot(lambdas, equations, label=f"(p1, p2) = ({self.p1}, {self.p2}), a = 1")
        plt.axhline(0, color='gray', lw=0.5, ls='--')

        if np.isfinite(self.lambda_star):
            plt.axvline(self.lambda_star, color="red", ls="--", lw=0.8)
        plt.plot(self.lambda_star, 0, 'ro', label=fr"$\lambda^* = {self.lambda_star:.4f}$")

        title = r"$F(\lambda) := \lambda u_1(\lambda) - 2u_0(\lambda) \ln u_0(\lambda) + \lambda u_0(\lambda)(p_2 - p_1) = 0$"
        plt.title(title)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$F(\lambda)$")
        plt.legend(loc="lower left")
        plt.grid()
        plt.show()

class SubGaussianBetaProxy:

    """
    Class for computing the optimal sub-Gaussian variance proxy for 
    Beta distributions 

    Attributes:
    ----------
    alpha : float
        The first shape parameter (must satisfy  alpha > 0).
    beta : float
        The second shape parameter (must satisfy beta > 0).

    Returns:
    -------
    sigma_opt_squared : float
        The computed optimal variance proxy (initialized after computation).
    lambda_star : float
        The optimal λ maximizing h(λ).
        
    """
    
    def __init__(self, alpha: float, beta: float):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Parameters must be positive")
        self.alpha = alpha
        self.beta = beta 
        self.variance = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        self.mu = self.alpha / (self.alpha + self.beta)
        self.sigma_opt_squared = None
        self.lambda_star = None

        self.bounds_list = [2, 5, 10, 20, 50, 100]  
        self.bracket_scales = [1, 2, 5, 10, 20, 40]  

    def h_beta(self, lam: float) -> float:
        """Compute h(λ) with safe fallback."""
        if abs(lam) < 1e-14:
            return self.variance

        try:
            val = np.exp(-lam * self.mu) * hyp1f1(self.alpha, self.alpha + self.beta, lam)
            if val > 0 and np.isfinite(val):
                result = 2.0 / (lam * lam) * np.log(val)
                if np.isfinite(result):
                    return result
        except Exception:
            pass
        
        return -np.inf  # Safe fallback for any error


    def plot_objective_function(self, n_points: int = 50000):
        """
        Plot h(λ) = 2/λ² * log E[exp(λ(X-μ))] and its maximum.
        """
        # Ensure we have computed the optimal values
        opt_val, lam_star = self.subgaussian_optimal_variance_proxy()
        
        if not np.isfinite(lam_star):
            print("Cannot plot: lambda_star is not finite.")
            return

        lam_vals = np.linspace(lam_star - 1, lam_star + 1, n_points)
        h_vals = [self.h_beta(l) for l in lam_vals]

        plt.figure(figsize=(8, 5))
        plt.plot(lam_vals, h_vals, label="h(λ)")
        plt.axvline(lam_star, color="gray", ls="--", label=f"λ*={lam_star:.2f}")
        if np.isfinite(lam_star):
            label_template = r"$\max_{{\lambda}} h={:.4f} \ at \ \lambda^*={:.2f}$"
            plt.scatter([lam_star], [opt_val], color="red", zorder=5,
            label=label_template.format(opt_val, lam_star))
        plt.xlabel("λ")
        plt.ylabel("h(λ)")
        plt.title(f"h(λ) for Beta(α={self.alpha}, β={self.beta})")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()    
            
    def subgaussian_optimal_variance_proxy(self):
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
            sigma_opt_squared = self.variance
            return sigma_opt_squared, lambda_star


        sigma_opt_squared = self.variance
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
                    self.sigma_opt_squared = -result.fun
                    self.lambda_star = float(result.x)
                    if self.sigma_opt_squared >= self.variance * 0.999:
                        return self.sigma_opt_squared, self.lambda_star
            except Exception:
                continue

        for bound in self.bounds_list:
            try:
                result = minimize_scalar(
                    lambda lam: -self.h_beta(lam),
                    bounds=(-bound, bound),
                    method='bounded'
                )
                if result.success and np.isfinite(result.fun):
                    optimal_value = -result.fun
                    if optimal_value >= self.variance * 0.999 and optimal_value > sigma_opt_squared:
                        self.sigma_opt_squared = optimal_value
                        self.lambda_star = float(result.x)
                        if abs(result.x) < 0.9 * bound:
                            break
            except Exception:
                continue  

        return self.sigma_opt_squared, self.lambda_star


class SubGaussianKumaraswamyProxy:

    """
    Class for computing the optimal sub-Gaussian variance proxy for 
    Kumaraswamy distributions 

    Attributes:
    ----------
    alpha : float
        The first shape parameter (must satisfy  alpha > 0).
    beta : float
        The second shape parameter (must satisfy beta > 0).

    Returns:
    -------
    sigma_opt_squared : float
        The computed optimal variance proxy (initialized after computation).
        
    lambda_star : float
        The optimal λ maximizing h(λ).
    
    """

    def __init__(self, alpha: float, beta: float):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Parameters must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)

        # E[X^r] = beta * B(1 + r/alpha, beta)
        self.mu = self._EX_pow_r(1.0)
        ex2 = self._EX_pow_r(2.0)
        self.var = ex2 - self.mu**2

        self.sigma_opt_squared = None
        self.lambda_star = None


        self.bounds_list = [2, 5, 10, 20, 50, 100]
        self.bracket_scales = [1, 2, 5, 10, 20, 40]

    def _EX_pow_r(self, r: float) -> float:
        # E[X^r] = beta * B(1 + r/alpha, beta)  (stable via logs)
        return np.exp(np.log(self.beta) + betaln(1.0 + r / self.alpha, self.beta))

    
    
    def _E_exp_lambda_X(self, lam: float, tol: float = 1e-12, max_terms: int = 100000) -> float:

        if np.isclose(self.alpha, 1.0):
            return float(hyp1f1(1.0, self.beta + 1.0, lam))

        a = self.alpha
        b = self.beta

        log_gamma_beta1 = gammaln(b + 1.0)

        total = 0.0
        k = 0
        # term_k = exp(log Γ(β+1) + log Γ(1+k/α) - log Γ(β+1+k/α) + k*log|lam| - log(k!))
        log_abs_lam = np.log(abs(lam)) if lam != 0.0 else -np.inf
        log_fact = 0.0  # cumul  log(k!) 

        prev_total = None
        while k < max_terms:
            if k > 0:
                log_fact += np.log(k)

            log_num = log_gamma_beta1 + gammaln(1.0 + k / a)
            log_den = gammaln(b + 1.0 + k / a) + log_fact
            if lam == 0.0 and k > 0:
                break
            log_term = log_num - log_den + (0.0 if lam == 0.0 else k * log_abs_lam)

            term = np.exp(log_term)
            if lam < 0.0 and (k % 2 == 1):
                term = -term

            new_total = total + term

            if total != 0.0 and abs(term) <= tol * abs(new_total):
                total = new_total
                break

            if prev_total is not None and abs(new_total - prev_total) <= max(tol * abs(new_total), 1e-18):
                total = new_total
                break

            prev_total = total
            total = new_total
            k += 1

        return total if np.isfinite(total) else np.nan
    

    def h_kumar(self, lam: float) -> float:
        """Compute h(λ) with safe fallbacks."""
        if abs(lam) < 1e-14:
            return self.var

        Ee_lamX = self._E_exp_lambda_X(lam)
        if not np.isfinite(Ee_lamX) or Ee_lamX <= 0.0:
            return -np.inf


        log_mgf_centered = -lam * self.mu + np.log(Ee_lamX)
        result = (2.0 / (lam * lam)) * log_mgf_centered
        return result if np.isfinite(result) else -np.inf

    def subgaussian_optimal_variance_proxy(self):
        """
        Adaptive search for σ²_opt = max_λ h(λ) with X ~ Kumaraswamy(α, β).
        Returns (sigma_opt_squared, lambda_star).
        """

        if np.isclose(self.alpha, 1.0) and np.isclose(self.beta, 1.0):
            self.sigma_opt_squared = self.var
            self.lambda_star = 0.0
            return self.sigma_opt_squared, self.lambda_star


        best_val = self.var
        best_lam = 0.0


        for scale in self.bracket_scales:
            try:
                bracket = (-scale, 0.0, scale)
                res = minimize_scalar(lambda lam: -self.h_kumar(lam), bracket=bracket, method='brent')
                if res.success and np.isfinite(res.fun):
                    val = -res.fun
                    lam_star = float(res.x)
                    if val > best_val:
                        best_val, best_lam = val, lam_star
                        if best_val >= self.var * 0.999:
                            self.sigma_opt_squared = best_val
                            self.lambda_star = best_lam
                            return self.sigma_opt_squared, self.lambda_star
            except Exception:
                continue


        for bound in self.bounds_list:
            try:
                res = minimize_scalar(lambda lam: -self.h_kumar(lam),
                                    bounds=(-bound, bound), method='bounded',
                                    options={"xatol": 1e-4})
                if res.success and np.isfinite(res.fun):
                    val = -res.fun
                    lam_star = float(res.x)
                    if val > best_val:
                        best_val, best_lam = val, lam_star
                        if abs(lam_star) < 0.9 * bound:
                            break
            except Exception:
                continue

        self.sigma_opt_squared = best_val
        self.lambda_star = best_lam
        return self.sigma_opt_squared, self.lambda_star


    def plot_objective_function(self, n_points: int = 50000):
        """
        Plot h(λ) and indicate the maximizer.
        """
        sigma_opt_squared, lam_star = self.subgaussian_optimal_variance_proxy()
        width = 1.0 if lam_star == 0 else max(1.0, 0.5 * (1.0 + abs(lam_star)))
        lam_vals = np.linspace(lam_star - width, lam_star + width, n_points)
        h_vals = [self.h_kumar(l) for l in lam_vals]

        plt.figure(figsize=(8, 5))
        plt.plot(lam_vals, h_vals, label="h(λ)")
        plt.axvline(lam_star, color="red", ls="--", lw=1.2, label=f"λ*={lam_star:.3g}")
        if np.isfinite(lam_star) and np.isfinite(sigma_opt_squared):
            plt.scatter([lam_star], [sigma_opt_squared], zorder=5, 
                        label=fr"$\max_\lambda h = {sigma_opt_squared:.4g}$ at $\lambda^*={lam_star:.3g}$")
            
        plt.xlabel("λ")
        plt.ylabel("h(λ)")
        plt.title(f"h(λ) for Kumaraswamy(α={self.alpha}, β={self.beta})")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()