import celmech as cm
import rebound as rb
import numpy as np
import sys
from construct_secular_matrices import get_solar_system_pvars, make_secular_matrices
from matplotlib import pyplot as plt
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
import re

ARCSEC_PER_YR = (np.pi / 180 /3600) / (2*np.pi)
to_array = lambda x: np.array(list(x))

# Generate and diagonalize Laplace-Lagrange matrices
pvars = get_solar_system_pvars()
Se,SI = make_secular_matrices(pvars)
De,Te = np.linalg.eigh(Se)
DI,TI = np.linalg.eigh(SI)
Te = Te[:,::-1] # put eccentricty modes in ascending order by magnitude of frequency
De = De[::-1]

# Compute the critical AMD of the terrestrial planets
from celmech.miscellaneous import critical_relative_AMD
alpha = pvars.particles[1].a/pvars.particles[2].a
gamma = pvars.particles[1].m/pvars.particles[2].m
AMD_crit = pvars.particles[2].Lambda * critical_relative_AMD(alpha,gamma)

# Read in data
infile = sys.argv[1] # pass input file as argument
outfile = re.sub("xy_data","uv_data",infile)
xdata = np.load(infile)

# compute complex secular eccentricity modes
u = Te.T @ xdata['x']

# Do FMFT on each mode in blocks of size 'block_size'
# Save the leading mode frequencies
G5 = 4.257506 * ARCSEC_PER_YR
TOL = 0.01
block_size = 512
times = xdata['times']
Nblocks = len(times) // block_size 
Nfreq = 6
T = times[:block_size]

# frequency and amplitdue of each free secular mode
freqs = np.zeros((4,Nblocks))
amps = np.zeros((4,Nblocks),dtype = np.complex128)

u_free = u.copy() # u with g5 forcing removed.

# loop over planets
for k in range(4):
    # loop over data blocks
    for i in range(Nblocks):
        z = u[k,i*block_size:(i+1)*block_size]
        fmft_result = fmft(T,z,Nfreq)
        freqs_i = to_array(fmft_result.keys())
        amps_i = to_array(fmft_result.values())
        df = np.abs(freqs_i/G5 - 1)
        i5 = np.argmin(df)
        if i5==0:
            i_mode = 1
        else:
            i_mode = 0 
        if df[i5]<TOL:
            u_free[k,i*block_size:(i+1)*block_size] -= amps_i[i5] * np.exp(1j * freqs_i[i5] * T)
        freqs[k,i] = freqs_i[i_mode]
        amps[k,i] = amps_i[i_mode]

t_mid = 0.5*np.array([times[i*block_size] + times[(i+1)*block_size-1] for i in range(Nblocks)])
np.savez_compressed(
    outfile,
    times = times,
    midtimes = t_mid,
    u = u,
    u_free = u_free,
    freqs = freqs,
    amps = amps
)
