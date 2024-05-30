import rebound as rb
import numpy as np
import celmech as cm
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
import sys

# Using version < 4?
OLD_REBOUND = True
in_file = sys.argv[1]
out_file = sys.argv[2]

if OLD_REBOUND:
    sa_function = rb.SimulationArchive
else:
    sa_function = rb.Simulationarchive


ARCSEC_PER_YR =  (3600 * 180 / np.pi)**(-1)
G5 = 4.257506 * ARCSEC_PER_YR
to_array = lambda x: np.array(list(x))
TOL = 1e-2

sa = sa_function(in_file)
output_dt = sa[1].t
Nout = 2**int(np.floor(np.log2(len(sa))))
Nskip = 2
Nsave = Nout // Nskip

# Fill array of complex eccentricity variables
x = np.zeros((4,Nsave),dtype = np.complex128)
times = np.zeros(Nsave)
for i in range(0,Nsave):
    sim = sa[i*Nskip]
    times[i] = sim.t
    cm.nbody_simulation_utilities.align_simulation(sim)
    pvars = cm.Poincare.from_Simulation(sim)
    ps = pvars.particles
    for j,p in enumerate(ps[1:5]):
        x[j,i] = p.x        

## Use FMFT to compute g5 forced components
# Block size for FMFT
Nblock = 2**int(np.ceil(np.log2(2*np.pi*1e7/output_dt))) // Nskip

# times to use in FMFT
T = times[:Nblock] / (2*np.pi)

# Array to store free x's 
x_free = x.copy()

# Number of frequencies to fit
Nfreq = 6

kmax= Nsave//Nblock - 1
for k in range(kmax):
    for j in range(4):
        fmft_result = fmft(
            T,
            x[j,k*Nblock:(k+1)*Nblock],
            Nfreq
        )
        freqs,amps = to_array(fmft_result.keys()),to_array(fmft_result.values())
        df = np.abs(freqs/G5 - 1)
        i5 = np.argmin(df)
        if df[i5]<TOL:
            amp = amps[i5]
            f5 = freqs[i5]
            xforced =  amp * np.exp(1j * f5 * T)
        else:
            xforced = 0
        x_free[j,k*Nblock:(k+1)*Nblock] -= xforced

np.savez_compressed(out_file,x=x,x_free = x_free)