import numpy as np
# Leg Position Constants
FRONT_LEFT = 0
FRONT_RIGHT = 1
BACK_LEFT = 2
BACK_RIGHT = 3

# Gait Constants
WALKING_TROT = 0
BOUND = 1
PACED = 2
GALLUP = 3
TROT_RUN= 4 
CRAWL = 5
STAND = 6

#
NUM_LEGS = 4

POSITION = 0
VELOCITY = 1
ACCELERATION = 2

NO_FORCE = 1

params = {}
params['L'] = 0.1934
params['W'] = 0.0465
params['d'] = 0.0955
params['knee'] = [-0.213, 0, 0]
params['foot'] = [-0.213, 0, 0.002]
params['sign_d'] = [1, -1, 1, -1]
params['sign_L'] = [1, 1, -1, -1]  
params['h'] = 0.27
params['g'] = 9.81
params['h_min'] = 0.0102145
params['h_max'] = 0.0602145
params['k_fst'] = 0.03
params['period'] = 0.3
params['t_mpc'] = 0.04
params['t_sim'] = 0.002
params['pred_horizon'] = 6
params['max_lin_acc'] = 0.8
params['In_b'] = [[ 2.4679290e-02,  0,  0],
                  [ 0,  1.0136107e-01, 0],
                  [ 0, 0,  1.1008498e-01]]

params['IKN_MAX_IT'] = 10
params['IKN_EPS'] = 1e-4
params['mass']= 15.2064
params['mu'] = 0.8
params['fmax'] = 200

params['Q'] = np.diag(np.array([1e5, 2e6, 1e6, 1e6, 1e6, 10e3, 1e3, 1e4, 1e3, 10, 10, 40, 0]))
params['R'] = np.diag(np.tile(np.array([0.1, 0.2, 0.1]), 4))
params['QT'] = np.diag(np.array([1e5, 2e6, 1e6, 1e6, 1e6, 10e3, 1e3, 1e4, 1e3, 10, 10, 40, 0]))
