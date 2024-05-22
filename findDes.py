import time

import numpy as np
import mujoco
import mujoco.viewer

from GaitDataLoader import GaitDataLoader
gait_data_loader = GaitDataLoader('FlatFoot4_data.mat')
q_actuated_des, dq_actuated_des, ddq_actuated_des = gait_data_loader.evaluate_bezier_curve(0.0)

q_i = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 14, 15, 16, 17, 18, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 50, 55, 56, 57, 58, 59, 60]
q_i = [x - 7 for x in q_i[7:]]

q_actuated = [7, 8, 9, 10, # left leg
            14, 15, # left toe A and B
            18, 19, 20, 21, # left hand
            22, 23, 24, 25, # right leg
            29, 30, # right toe A and B
            33, 34, 35, 36] # right hand
q_actuated = [x - 7 for x in q_actuated]

m = mujoco.MjModel.from_xml_path('models/digit-v3.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()
    
    q_pos = d.qpos[q_i]
    ctrl = 5*(q_actuated_des - q_pos[q_actuated])
    d.ctrl[:] = ctrl
    
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d, 15)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)