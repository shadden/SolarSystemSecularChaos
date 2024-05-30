import celmech as cm
import numpy as np
import rebound as rb
from celmech.disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict

def get_solar_system_sim():
    sim = rb.Simulation()
    sim.add("solar system")
    sim.move_to_com()
    cm.nbody_simulation_utilities.align_simulation(sim)
    return sim

def get_solar_system_pvars():
    sim = get_solar_system_sim()
    pvars = cm.Poincare.from_Simulation(sim)
    return pvars

def make_secular_matrices(pvars):
    C_ein_sq = df_coefficient_C(*(0 for _ in range(6)),0,0,1,0)
    C_eout_sq = df_coefficient_C(*(0 for _ in range(6)),0,0,0,1)
    C_ein_eout = df_coefficient_C(*[0,0,1,-1,0,0],0,0,0,0)
    
    C_Iin_sq = df_coefficient_C(*(0 for _ in range(6)),1,0,0,0)
    C_Iout_sq = df_coefficient_C(*(0 for _ in range(6)),0,1,0,0)
    C_Iin_Iout = df_coefficient_C(*[0,0,0,0,1,-1],0,0,0,0)

    Se = np.zeros((4,4))
    SI = np.zeros((4,4))
    ps = pvars.particles[1:]
    GG = pvars.G
    for i in range(4):
        Lambda_i = ps[i].Lambda
        a_i = ps[i].a
        m_i = ps[i].m
        f_i = np.sqrt(1/Lambda_i)
        for j in range(i+1,4):
            Lambda_j = ps[j].Lambda
            a_j = ps[j].a
            m_j = ps[j].m
            f_j = np.sqrt(1/Lambda_j)

            alpha = a_i / a_j
            prefactor = -GG * m_i * m_j  / a_j
            
            Cij = evaluate_df_coefficient_dict(C_ein_eout,alpha)
            Cii = evaluate_df_coefficient_dict(C_ein_sq,alpha)
            Cjj = evaluate_df_coefficient_dict(C_eout_sq,alpha)
            Se[i,j] +=  prefactor * f_i * f_j * Cij
            Se[j,i] +=   prefactor * f_i * f_j * Cij             
            Se[i,i] +=  2 * prefactor * f_i * f_i * Cii
            Se[j,j] +=  2 * prefactor * f_j * f_j * Cjj
            
            Cij = evaluate_df_coefficient_dict(C_Iin_Iout,alpha)
            Cii = evaluate_df_coefficient_dict(C_Iin_sq,alpha)
            Cjj = evaluate_df_coefficient_dict(C_Iout_sq,alpha)
            SI[i,j] +=  prefactor * f_i * f_j * Cij
            SI[j,i] +=   prefactor * f_i * f_j * Cij             
            SI[i,i] +=  2 * prefactor * f_i * f_i * Cii
            SI[j,j] +=  2 * prefactor * f_j * f_j * Cjj
        for j in range(4,8):
            Lambda_j = ps[j].Lambda
            a_j = ps[j].a
            m_j = ps[j].m
            f_j = np.sqrt(1/Lambda_j)
            alpha = a_i / a_j
            prefactor = -GG * m_i * m_j  / a_j

            Cii = evaluate_df_coefficient_dict(C_ein_sq,alpha)
            Se[i,i] +=  2 * prefactor * f_i * f_i * Cii

            Cii = evaluate_df_coefficient_dict(C_Iin_sq,alpha)
            SI[i,i] +=  2 * prefactor * f_i * f_i * Cii

    SI *= 0.25

    return Se, SI

def compute_u_forced(pvars,giant_mode_frequencies,giant_mode_amplitudes):
    pass