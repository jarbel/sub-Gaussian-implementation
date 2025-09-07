
![MIT License](https://img.shields.io/badge/license-MIT-green.svg)

# Sub-Gaussian Proxy Variance Library

This library provides tools to compute the **optimal sub-Gaussian variance proxy** for a variety of discrete and continuous probability distributions. It implements advanced algorithms for strict sub-Gaussianity, optimal proxy variance, and related properties, as described in recent probability literature.

## Features

- **Bernoulli and Binomial**: Compute the optimal sub-Gaussian variance proxy for Bernoulli and Binomial distributions.
- **Uniform (Continuous and Discrete)**: Compute the proxy for uniform distributions, including sums of independent uniforms.
- **Truncated Normal**: Compute the proxy for truncated normal distributions.
- **Triangular Distribution**: Advanced class for strict sub-Gaussian proxy variance, including numerical optimization and plotting.
- **3-Mass Discrete Distributions**: Classes for symmetric and asymmetric 3-mass distributions.
- **Beta and Kumaraswamy Distributions**: Compute the proxy for Beta and Kumaraswamy distributions, with adaptive optimization.


## Installation

This library requires **Python 3.9 or higher**.

```bash
pip install -r requirements.txt
```

## Usage

Import the relevant functions or classes from `variance_proxy.py`:

```python
from source.variance_proxy import (
    subgaussian_proxy_variance_bernoulli,
    subgaussian_proxy_variance_binomial,
    subgaussian_proxy_variance_uniform,
    SubGaussianTriangularProxy,
    SubGaussian3MassSymetricProxy,
    SubGaussian3MassAssymetricProxy,
    SubGaussianBetaProxy,
)
```

### Example: Bernoulli Proxy

```python
sigma_opt_squared = subgaussian_proxy_variance_bernoulli(0.3)
print(f"Optimal proxy variance: {sigma_opt_squared}")
```

### Example: Triangular Distribution

```python
tri = SubGaussianTriangularProxy(a=1, b=2)
sigma_opt_squared = tri.subgaussian_optimal_variance_proxy()
print(f"Optimal proxy variance: {sigma_opt_squared}")
```

## API Overview

- **subgaussian_proxy_variance_bernoulli(p)**: Optimal proxy for Bernoulli(p).
- **subgaussian_proxy_variance_binomial(n, p)**: Optimal proxy for Binomial(n, p).
- **subgaussian_proxy_variance_uniform(a, b)**: Proxy for Uniform(a, b).
- **subgaussian_proxy_variance_sum_independant_uniform(segments)**: Proxy for sum of independent uniforms.
- **subgaussian_discrete_uniforme_variance_proxy(a, n)**: Proxy for discrete uniform.
- **subgaussian_proxy_variance_truncated_normal(a, b, mu, sigma2)**: Proxy for truncated normal.
- **SubGaussianTriangularProxy(a, b)**: Class for triangular distribution.
- **SubGaussian3MassSymetricProxy(p, a)**: Class for symmetric 3-mass distribution.
- **SubGaussian3MassAssymetricProxy(p1, p2, a)**: Class for asymmetric 3-mass distribution.
- **SubGaussianBetaProxy(alpha, beta)**: Class for Beta distribution.
- **SubGaussianKumaraswamyProxy(alpha, beta)**: Class for Kumaraswamy distribution.

## References and Citations

If you use this library in your research, please cite:

- Barreto, M., Marchal, O., & Arbel, J. (2024). *Optimal sub-Gaussian variance proxy for truncated Gaussian and exponential random variables*.  
  *arXiv preprint*: [arXiv:2403.08628](https://arxiv.org/abs/2403.08628)

- Arbel, J., Marchal, O., & Nguyen, H. D. (2020). *On strict sub-Gaussianity, optimal proxy variance and symmetry for bounded random variables*.  
  ESAIM: Probability & Statistics, 24, 39–55.  
  *arXiv preprint*: [arXiv:1901.09188](https://arxiv.org/abs/1901.09188)

- Marchal, O., & Arbel, J. (2017). *On the sub-Gaussianity of the Beta and Dirichlet distributions*.  
  *arXiv preprint*: [arXiv:1705.06197](https://arxiv.org/abs/1705.06197)


## Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request.