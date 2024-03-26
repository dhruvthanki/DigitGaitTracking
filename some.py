import time

import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('models/digit-v3.xml')
d = mujoco.MjData(m)
d.qpos[:] = m.key_qpos
mujoco.mj_forward(m, d)

q_i = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 14, 15, 16, 17, 18, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 50, 55, 56, 57, 58, 59, 60]
q_i = [x - 7 for x in q_i[7:]]

q_actuated = [7, 8, 9, 10, # left leg
            14, 15, # left toe A and B
            18, 19, 20, 21, # left hand
            22, 23, 24, 25, # right leg
            29, 30, # right toe A and B
            33, 34, 35, 36] # right hand
q_actuated = [x - 7 for x in q_actuated]

q1 = 1.0*np.array([0.37515,-0.00010001,0.42401,0.08085,0.40686,-0.3818,])  # left leg
q2 = 1.0*np.array([-0.16236,0.37278,-0.91346,1.2283,]) # left hand
q3 = 1.0*np.array([-0.37515,0.00010001,0.020586,-0.017141,-0.36649,0.45382,]) # right leg
q4 = 1.0*np.array([-0.21256,-1.6102,0.1011,-0.42019]) # right hand
des_pos = np.concatenate([q1, q2, q3, q4])

des_pos = np.array([3.75145791e-01, -1.00009930e-04,  4.24013896e-01,  8.08496532e-02,
        4.06861475e-01, -3.81802977e-01, -1.62356150e-01,  3.72775628e-01,
       -9.13455247e-01,  1.22834364e+00, -3.75145790e-01,  1.00009949e-04,
        2.05856025e-02, -1.71408981e-02, -3.66485506e-01,  4.53821183e-01,
       -2.12555135e-01, -1.61019775e+00,  1.01099878e-01, -4.20185338e-01])

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  while viewer.is_running():
    step_start = time.time()
    
    q_pos = d.qpos[q_i]
    ctrl = 40*(des_pos - q_pos[q_actuated])
    ctrl[[4, 5, 14, 15]] = 0
    # ctrl[[3, 13]] = 0
    d.ctrl[:] = ctrl
    
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)
        
    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)