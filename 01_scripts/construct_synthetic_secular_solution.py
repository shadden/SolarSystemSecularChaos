import numpy as np
import rebound as rb
import reboundx as rbx
from reboundx.constants import C as _SPEED_OF_LIGHT
import celmech as cm
from celmech.nbody_simulation_utilities import get_simarchive_integration_results as get_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft

ARCSEC_PER_YR =  (3600 * 180 / np.pi)**(-1)

#_SPEED_OF_LIGHT_AU_PER_YEAR = 63198
def get_solar_system():
    sim = rb.Simulation()
    #sim.units=('au','Msun','yr')
    sim.add("solar system")
    sim.move_to_com()
    cm.nbody_simulation_utilities.align_simulation(sim)
    rebx = rbx.Extras(sim)
    gr = rebx.load_force("gr_potential")
    gr.params["c"] = _SPEED_OF_LIGHT
    rebx.add_force(gr)
    sim.integrator = 'whckl'
    sim.dt = (4./365.) * 2 * np.pi
    sim.ri_whfast.safe_mode = 0                 # combines symplectic correctors and drift steps at beginning and end of timesteps.
    sim.ri_whfast.keep_unsynchronized = True    # allows for bit-by-bit reproducibility
    return sim

def get_solar_system_J2():
    sim = rb.Simulation()
    dummy_sim = get_solar_system()
    Mc = np.sum([p.m for p in dummy_sim.particles[:5]])
    sim.add(m=Mc)
    
    bodies = ["Jupiter","Saturn","Uranus","Neptune"]
    for i in range(5,9):
        sim.add(dummy_sim.particles[i].copy())
    sim.move_to_com()
    cm.nbody_simulation_utilities.align_simulation(sim)
    rebx = rbx.Extras(sim)
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    sun = sim.particles[0]

    Req = 1/200.
    J2 = 0.5 * np.sum([(p.m/Mc) * (p.a/Req)**2  for p in dummy_sim.particles[1:5]])
    J4 = (-3/8) * np.sum([(p.m/Mc) * (p.a/Req)**4  for p in dummy_sim.particles[1:5]])
    sun.params["J2"] = J2
    sun.params["J4"] = J4
    sun.params["R_eq"] = Req
    sim.integrator = 'whfast'
    sim.dt = sim.particles[1].P / 25.0
    sim.ri_whfast.safe_mode = 0                 # combines symplectic correctors and drift steps at beginning and end of timesteps.
    sim.ri_whfast.keep_unsynchronized = True    # allows for bit-by-bit reproducibility
    return sim

if __name__=="__main__":
    save_file = "../03_data/outer_solar_system.sa"
    try:
        sa = rb.Simulationarchive(save_file)
    except:
        print("Loading simulation archive '{}' failed".format(save_file))
        print("Running integration...")
        ssj2_sim = get_solar_system_J2()
        steps = int(2*np.pi*1e3//ssj2_sim.dt)
        step_size = (2*np.pi*1e3//ssj2_sim.dt) * ssj2_sim.dt
        pow2 = np.ceil(np.log2(2*np.pi*3e6/step_size))
        Tfin = 2**pow2 * step_size
        ssj2_sim.save_to_file(save_file,step=steps)
        ssj2_sim.integrate(Tfin)
        sa = rb.Simulationarchive(save_file)

    # Run FMFT on complex eccentricities
    results = get_results(sa,coordinates='heliocentric')
    results['z'] = results['e'] * np.exp(1j * results['pomega'])
    T = results['time']/(2*np.pi)
    fmftresults_ecc = dict()
    for i in range(5,9):
        print(i)
        print("---------------")
        z = results['z'][i-5]
        fmftresults_ecc[i] = fmft(T,z,4,min_freq=0.1*ARCSEC_PER_YR,max_freq=35*ARCSEC_PER_YR)
        for f, amp in fmftresults_ecc[i].items():
            print("{:.3f} \t {:.5f}".format(f / ARCSEC_PER_YR,np.abs(amp)))
        print("")

    # Construct S_e matrix
    to_array = lambda x: np.array(list(x))
    TOL = 1e-2
    g_freqs = np.array([list(fmftresults_ecc[i].keys())[0] for i in range(5,9)])
    g_phases = np.array([np.angle(list(fmftresults_ecc[i].values()))[0] for i in range(5,9)])
    Smtrx_ecc = np.zeros((4,4))
    for i in range(4):
        f = g_freqs[i]
        phi = g_phases[i]
        for j in range(4):
            freqs,amps = to_array(fmftresults_ecc[j+5].keys()),to_array(fmftresults_ecc[j+5].values())
            df = np.abs(freqs/f-1)
            imode = np.argmin(df)
            if df[imode]<TOL:
                amp,angle = np.abs(amps[imode]),np.angle(amps[imode])
                Smtrx_ecc[j,i] = amp * np.sign(np.cos(angle-phi))        

    # Run FMFT on complex inclinations
    results['zeta'] = np.sin(0.5*results['inc']) * np.exp(1j * results['Omega'])
    fmftresults_inc = dict()
    for i in range(5,9):
        print(i)
        print("---------------")
        zeta = results['zeta'][i-5]
        fmftresults_inc[i] = fmft(T,zeta,3,min_freq=-35*ARCSEC_PER_YR,max_freq=-0.1*ARCSEC_PER_YR)
        for f, amp in fmftresults_inc[i].items():
            print("{:.3f} \t {:.5f}".format(f / ARCSEC_PER_YR,np.abs(amp)))
        print("")

    # Construct S_inc matrix
    s_freqs = list(fmftresults_inc[5].keys())
    s_phases = np.angle(list(fmftresults_inc[5].values()))
    Smtrx_inc = np.zeros((4,3))
    for i in range(3):
        f = s_freqs[i]
        phi = s_phases[i]
        for j in range(4):
            freqs,amps = to_array(fmftresults_inc[j+5].keys()),to_array(fmftresults_inc[j+5].values())
            df = np.abs(freqs/f-1)
            imode = np.argmin(df)
            if df[imode]<TOL:
                amp,angle = np.abs(amps[imode]),np.angle(amps[imode])
                Smtrx_inc[j,i] = amp * np.sign(np.cos(angle-phi))        
    

    np.savez_compressed("../03_data/outer_solar_system_synthetic_secular_solution",
                        Smtrx_ecc = Smtrx_ecc,
                        Smtrx_inc = Smtrx_inc,
                        g_freqs = g_freqs,
                        s_freqs = s_freqs,
                        g_phases = g_phases,
                        s_phases =s_phases
                        )