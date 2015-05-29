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

cdef inline double ln_laplace(double value, double mean, double stddev):
    return -1.0 * fabs(value - mean) / stddev - log(stddev)

cdef inline double ln_gamma(double value, double k, double theta):
    return (k - 1.0) * log(value) - value / theta

cdef inline double ln_beta(double value, double p, double q):
    return (p - 1.0) * log(value) + (q - 1.0) * log(1 - value)

cdef inline double normal(double value, double mean, double stddev):
    cdef double tmp = (value - mean) / stddev
    tmp *= tmp
    return exp(-0.5 * tmp) / stddev
