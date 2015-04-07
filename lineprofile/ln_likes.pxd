"A bunch of improper (not normalized) prior distributions"
from libc.math cimport log, fabs, sqrt, exp, M_PI

cdef inline double ln_normal(double value, double mean, double stddev):
    cdef double tmp = (value - mean) / stddev
    tmp *= tmp
    return -0.5 * tmp - log(stddev)

cdef inline double ln_laplace(double value, double mean, double stddev):
    return -1.0 * fabs(value - mean) / stddev - log(stddev)

cdef inline double ln_log(double value):
    return -1.0 * log(value)

cdef inline double ln_gamma(double value, double k, double theta):
    return (k - 1.0) * log(value) - value / theta

cdef inline double ln_beta(double value, double p, double q):
    return (p - 1.0) * log(value) + (q - 1.0) * log(1 - value)

cdef inline double normal(double value, double mean, double stddev):
    cdef double tmp = (value - mean) / stddev
    tmp *= tmp
    return exp(-0.5 * tmp) / sqrt(2. * M_PI) / stddev
