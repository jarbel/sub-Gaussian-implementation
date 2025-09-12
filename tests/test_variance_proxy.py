"""
How to run the tests:
-----------------------
>> python -m tests.test_variance_proxy
or 
VS code debugger (launch json config)
"""
from scipy.optimize import root_scalar
from src.variance_proxy import SubGaussian3MassAsymmetricProxy, SubGaussian3MassSymmetricProxy, subgaussian_proxy_variance_bernoulli, SubGaussianBetaProxy
import matplotlib.pyplot as plt 
import numpy as np


check_asymmetric_symmetric_implementation = True
replicate_paper_beta_bernoulli_figs = False
test_3mass_sym_and_assym = False

if check_asymmetric_symmetric_implementation:

    p_values = np.linspace(0.0001, 1/6 - 0.0001, 3000)
    proxy_variances_ass = []
    proxy_variances_s = []
    for p in p_values:
        objAssym = SubGaussian3MassAsymmetricProxy(p1=p, p2=p, a=1)
        sigma_opt_squared, _ = objAssym.subgaussian_optimal_variance_proxy()
        proxy_variances_ass.append(sigma_opt_squared)
        
        objSym = SubGaussian3MassSymmetricProxy(p)
        sigma_opt_squared, _ = objSym.subgaussian_optimal_variance_proxy()
        proxy_variances_s.append(sigma_opt_squared)

    proxy_variances_ass = np.array(proxy_variances_ass)
    proxy_variances_s = np.array(proxy_variances_s)


    rtol, atol = 1e-10, 1e-12
    ok = np.allclose(proxy_variances_ass, proxy_variances_s, rtol=rtol, atol=atol)
    print(f"Numerical equivalence (rtol={rtol}, atol={atol}):", ok)


    abs_err = np.abs(proxy_variances_ass - proxy_variances_s)
    print("Max absolute error:", abs_err.max())

    
    plt.figure()
    plt.plot(p_values, proxy_variances_s, linestyle='dashed', label="Symmetric", linewidth=2)
    plt.plot(p_values, proxy_variances_ass, label="Asymmetric")
    plt.title("Equivalence Check: Optimal Proxy Variance — Symmetric vs. Asymmetric")
    plt.xlabel("Probability (p < 1/6)")
    plt.ylabel("Optimal Proxy Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()


if replicate_paper_beta_bernoulli_figs:
    """
        This test reproduces the main figures from the paper:
        "On the sub-Gaussianity of the Beta and Dirichlet distributions" (2017).
        It validates that the implementation matches the theoretical
        results and visualizations (see Figure 1 in the paper).
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    mu = np.linspace(0.000001, 0.999999, 2000)
    S = 1.0  # fixed α + β
    alphas = mu * S
    betas = (1 - mu) * S

    variances = (alphas * betas) / (S**2 * (S + 1))  # Var[Beta] for α+β=1
    upper_bounds = [1 / (4 * (a + b + 1)) for a, b in zip(alphas, betas)]
    beta_sigma_opts = []
    bernoulli_sigma_opts = []

    for a, b in zip(alphas, betas):
        obj = SubGaussianBetaProxy(a, b)
        sigma_opt_squared, _ = obj.subgaussian_optimal_variance_proxy()
        beta_sigma_opts.append(sigma_opt_squared)
        bernoulli_sigma_opts.append(subgaussian_proxy_variance_bernoulli(a))

    plt.figure(figsize=(10, 6))
    plt.plot(mu, variances, 'g-', label='Variance', linewidth=2)
    plt.plot(mu, beta_sigma_opts, 'purple', label=r'$\sigma^2_{\mathrm{opt}}(Beta(\alpha,\beta))$', linewidth=2)
    plt.plot(mu, bernoulli_sigma_opts, 'b', label=r'$\sigma^2_{\mathrm{opt}}(Bern(\mu))$', linewidth=2)
    plt.plot(mu, upper_bounds, 'k:', label='Upper Bound $\\frac{1}{4(α+β+1)}$', linewidth=2)

    plt.axvline(0.5, color='gray', linestyle=':', linewidth=1)
    plt.xlabel(r'$\theta = \frac{\alpha}{\alpha + \beta}$', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.title(r'Variance and Optimal Sub-Gaussian Proxy for Beta($\alpha,\beta$) with $\alpha + \beta = 1$', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    S_vals = np.logspace(-1, 1, 14) 

    cmap = LinearSegmentedColormap.from_list("purple_red", ["#6A00A8", "#D00000"])
    norm = LogNorm(vmin=S_vals.min(), vmax=S_vals.max())

    plt.figure(figsize=(9.5, 6.2))

    bern = [subgaussian_proxy_variance_bernoulli(m) for m in mu]
    plt.plot(mu, bern, lw=2.5, label=r'$\sigma^2_{\mathrm{opt}}(\mathrm{Bern}(\mu))$', color='tab:blue')
    for S in S_vals:
        alphas = mu * S
        betas = (1 - mu) * S
        sigma2 = []
        for a, b in zip(alphas, betas):
            obj = SubGaussianBetaProxy(a, b)
            sigma_opt_squared, _ = obj.subgaussian_optimal_variance_proxy()
            sigma2.append(sigma_opt_squared)
        plt.plot(mu, sigma2, lw=2, color=cmap(norm(S)))


    plt.xlabel(r'$\mu=\frac{\alpha}{\alpha+\beta}$', fontsize=12)
    plt.ylabel(r'$\sigma^2_{\mathrm{opt}}$', fontsize=12)
    plt.title(r'Center: $\sigma^2_{\mathrm{opt}}(\mu)$ for Bernoulli (blue) and Beta with $S=\alpha+\beta\in[0.1,10]$ (purple $\to$ red)', fontsize=13)
    plt.legend(fontsize=10, loc='upper center')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




if test_3mass_sym_and_assym:

    #################################################### Compare Symmetric and Asymmetric implementations ####################################################
    p_values = np.linspace(0.0001, 1/6 - 0.0001, 500)  # p < 1/6
    proxy_variances_asym = []
    proxy_variances_sym = []
    for p in p_values:
        objAsym = SubGaussian3MassAsymmetricProxy(p1=p, p2=p, a=1)
        sigma2_opt_asym, _ = objAsym.subgaussian_optimal_variance_proxy()
        proxy_variances_asym.append(sigma2_opt_asym)
        
        objSym = SubGaussian3MassSymmetricProxy(p)
        sigma_opt_squared, _ = objSym.subgaussian_optimal_variance_proxy()
        proxy_variances_sym.append(sigma_opt_squared)

    plt.figure(figsize=(8, 6))
    plt.plot(p_values, proxy_variances_sym, label="symmetric", linestyle='dashed', linewidth=2)
    plt.plot(p_values, proxy_variances_asym, label="asymmetric")
    plt.title('Equivalence Check: Optimal Proxy Variance — Symmetric vs. Asymmetric Implementations')
    plt.xlabel('Probability (p<1/6)')
    plt.ylabel('Optimal proxy variance')
    plt.legend()
    plt.grid(True)
    plt.show()

    #################################################### Reproduction of paper plot ####################################################*

        
    tol = 1e-7
    sigma_opt_squared_values = []
    sigma_up_values = []
    sigma_2p_values = []

    for p in p_values:
        obj = SubGaussian3MassSymmetricProxy(p)
        sigma_opt_squared, _ = obj.subgaussian_optimal_variance_proxy()
        sigma_opt_squared_values.append(sigma_opt_squared)
        sigma_up_values.append(obj.upper_bound)
        sigma_2p_values.append(2 * p)

    plt.figure(figsize=(8, 6))
    plt.plot(p_values, sigma_2p_values, color='blue', label=r'$2p$')
    plt.plot(p_values, sigma_up_values, color='r', label=r'$\sigma_{\mathrm{up}}^2 = \frac{(1 - 2p)^2}{4(1 - 4p)}$')
    plt.plot(p_values, sigma_opt_squared_values, 'ko', markersize=3, label=r'$\sigma_{\mathrm{opt}}^2$')
    plt.xlabel(r'$p$')
    plt.ylabel(r'Variance proxy')
    plt.title('Optimal variance proxy for symmetric 3-mass discrete distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

    #################################################### Exhaustive test for asymmetric implementation ####################################################

    # success rate 1.0, total pairs 996004, fail 0
    s = 0
    f = 0

    for p1 in np.arange(0.001, 1-0.001, 0.001):
        for p2 in np.arange(0.001, 1-0.001, 0.001):
            s+=1
            if (p2 >= p1 and p1+p2<1):
                obj = SubGaussian3MassAsymmetricProxy(p1, p2, a=1)
                sigma2_opt, _  = obj.subgaussian_optimal_variance_proxy()
                if np.isnan(sigma2_opt):
                    f+=1
                    print(p1, p2, sigma2_opt)
    print(f'success rate {(s-f)/s}, total pairs {s}, fail {f}')
