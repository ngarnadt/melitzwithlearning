import glob
import os
import platform
 
from cffi import FFI

os.chdir(os.path.join('C:\\','Users','ngarnadt','Documents','Startups','Simulation','melitzwithlearning')) 
 
include_dirs = [os.path.join('externals', 'cquadpack', 'src'),
                os.path.join('externals', 'cquadpack', 'include')]
 
cquadpack_src = glob.glob(os.path.join('externals', 'cquadpack', 'src', '*.c'))
 
# Take out dSFMT dependant files; Just use the basic rng
cquadpack_src = [f for f in cquadpack_src if ('librandom.c' not in f) and ('randmtzig.c' not in f)]
 
extra_compile_args = ['-DMATHLIB_STANDALONE']
if platform.system() == 'Windows':
    extra_compile_args.append('-std=c99')
 
ffi = FFI()
ffi.set_source('_cquadpack', '#include <cquadpack.h>',
        include_dirs=include_dirs,
        sources=cquadpack_src,
        libraries=[],
        extra_compile_args=extra_compile_args)
 
# This is an incomplete list of the available functions in Rmath
# but these are sufficient for our example purposes and gives a sense of
# the types of functions we can get
ffi.cdef('''\
// Normal Distribution
double dnorm(double, double, double, int);
double pnorm(double, double, double, int, int);
 
// Uniform Distribution
double dunif(double, double, double, int);
double punif(double, double, double, int, int);
 
// Gamma Distribution
double dgamma(double, double, double, int);
double pgamma(double, double, double, int, int);
''')
 
if __name__ == '__main__':
    # Normally set verbose to `True`, but silence output
    # for reduced notebook noise
    ffi.compile(verbose=False)