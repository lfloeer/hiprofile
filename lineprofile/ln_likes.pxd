"A bunch of improper (not normalized) prior distributions"
from libc.math cimport log, fabs

cdef inline double ln_normal(double value, double mean, double stddev):
    cdef double tmp = (value - mean) / stddev
    tmp *= tmp
    return -0.5 * tmp - log(stddev)

cdef inline double ln_laplace(double value, double mean, double stddev):
    return -0.5 * fabs(value - mean) / stddev

cdef inline double ln_log(double value):
    return 1. / value

cdef inline double ln_gamma(double value, double k, double theta):
    return (k - 1.0) * log(value) - value / theta

cdef inline double ln_beta(double value, double p, double q):
    return (p - 1.0) * log(value) + (q - 1.0) * log(1 - value)