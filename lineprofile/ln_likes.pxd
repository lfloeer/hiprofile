"""
A bunch of improper (not normalized) prior distributions and their normalizations.
All functions are structured in a way that ln_* + ln_*_norm gives the normalized
logarithmic likelihood.
"""
from libc.math cimport log, fabs, sqrt, exp, M_PI, lgamma

cdef inline double ln_normal(double value, double mean, double stddev):
    cdef double tmp = (value - mean) / stddev
    tmp *= tmp
    return -0.5 * tmp - log(stddev)

cdef inline double ln_normal_norm():
    return -0.5 * log(2. * M_PI)

cdef inline double ln_laplace(double value, double mean, double stddev):
    return -1.0 * fabs(value - mean) / stddev - log(stddev)

cdef inline double ln_laplace_norm():
    return -1.0 * log(2.0)

cdef inline double ln_uniform_norm(double lower, double upper):
    return -1.0 * log(upper - lower)

cdef inline double ln_gamma(double value, double k, double theta):
    return (k - 1.0) * log(value) - value / theta

cdef inline double ln_gamma_norm(double k, double theta):
    return -1.0 * k * log(theta) - lgamma(k)

cdef inline double ln_beta(double value, double p, double q):
    return (p - 1.0) * log(value) + (q - 1.0) * log(1 - value)

cdef inline double ln_beta_norm(double p, double q):
    return lgamma(p + q) - lgamma(p) - lgamma(q)

cdef inline double normal(double value, double mean, double stddev):
    cdef double tmp = (value - mean) / stddev
    tmp *= tmp
    return exp(-0.5 * tmp) / stddev
