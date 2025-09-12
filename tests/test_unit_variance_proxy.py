"""
How to run these tests:
-----------------------
>> python -m pytest -v
or 
VS code debugger (launch json config)

Requires: pytest (pip install pytest)
"""

import pytest
import numpy as np
from src.variance_proxy import (
    subgaussian_proxy_variance_bernoulli,
    subgaussian_proxy_variance_binomial,
    subgaussian_proxy_variance_uniform,
    SubGaussianTriangularProxy,
    SubGaussian3MassSymmetricProxy,
    SubGaussian3MassAsymmetricProxy,
    SubGaussianBetaProxy,
)


def test_bernoulli_valid():
    assert np.isclose(subgaussian_proxy_variance_bernoulli(0.5), 0.25)
    assert np.isclose(subgaussian_proxy_variance_bernoulli(0.1), (0.5-0.1)/np.log(1/0.1-1))

def test_bernoulli_invalid():
    with pytest.raises(ValueError):
        subgaussian_proxy_variance_bernoulli(-0.1)
    with pytest.raises(ValueError):
        subgaussian_proxy_variance_bernoulli(1.1)

def test_binomial_proxy():
    assert np.isclose(subgaussian_proxy_variance_binomial(10, 0.5), 2.5)
    assert np.isclose(subgaussian_proxy_variance_binomial(0, 0.5), 0.0)

def test_uniform_proxy():
    assert np.isclose(subgaussian_proxy_variance_uniform(0, 1), 1/12)
    with pytest.raises(ValueError):
        subgaussian_proxy_variance_uniform(2, 1)

def test_triangular_variance_bounds():
    tri = SubGaussianTriangularProxy(1, 2)
    result = tri.subgaussian_optimal_variance_proxy()
    assert tri.variance <= result["sigma_opt_squared"] <= tri.upper

def test_triangular_symmetry():
    tri = SubGaussianTriangularProxy(2, 2)
    result = tri.subgaussian_optimal_variance_proxy()
    assert np.isclose(result["sigma_opt_squared"], tri.variance)
    assert np.isclose(result["lambda_opt"], 0.0)

def test_3mass_symmetric_proxy():
    obj = SubGaussian3MassSymmetricProxy(0.1)
    sigma2, lam = obj.subgaussian_optimal_variance_proxy()
    assert sigma2 > 0

def test_3mass_asymmetric_proxy():
    obj = SubGaussian3MassAsymmetricProxy(0.1, 0.2, 1)
    sigma2, lam = obj.subgaussian_optimal_variance_proxy()
    assert sigma2 > 0

def test_beta_proxy():
    obj = SubGaussianBetaProxy(2, 3)
    sigma2, lam = obj.subgaussian_optimal_variance_proxy()
    assert sigma2 > 0
