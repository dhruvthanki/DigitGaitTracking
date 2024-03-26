import time

import keyboard
import numpy as np
import mujoco
import mujoco.viewer
from transitions import Machine

from cWBQP2 import PWQP, ControllerStance
from GaitDataLoader import GaitDataLoader

class DigitSimulation:
    STANDING = np.array([0.0925871, 0.00188222, 0.967813, 0.999743, -0.000163863, 0.022431, 0.00340185, 0.331981, -0.00526764, 0.176355, 0.996243, -0.00335789, 0.000744049, 0.086534, 0.186855, -0.0140027, -0.137506, -0.0115589, -0.0529604, 0.974014, 0.224951, 0.00557492, 0.0257316, 0.0505937, 0.744686, -0.667007, 0.0160935, -0.0169332, 0.072832, -0.0478979, -0.106145, 0.894838, -0.00278591, 0.344714, -0.331981, 0.00526582, -0.176342, 0.996239, 0.00337517, 0.000743736, -0.0865791, -0.186917, 0.0139709, 0.137658, 0.0115404, 0.0529809, 0.974041, -0.224834, 0.00558042, -0.0257484, -0.0505853, 0.744537, 0.667173, 0.0160876, 0.0169263, -0.0728824, 0.0482856, 0.106047, -0.894876, 0.00300412, -0.344657])
    MODEL_FILE = 'models/digit-v3.xml'
    DATA_FILE = 'Digit-data_12699.mat'
    q_actuated = [7, 8, 9, 10, # left leg
            14, 15, # left toe A and B
            18, 19, 20, 21, # left hand
            22, 23, 24, 25, # right leg
            29, 30, # right toe A and B
            33, 34, 35, 36] # right hand
    
    def __init__(self):
        self.unpause = False
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_FILE)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 5
        self.viewer.cam.azimuth = 120
        self.gait_data_loader = GaitDataLoader(self.DATA_FILE)
        self.initialize_simulation()
    
    def initialize_simulation(self):
        self.t_step = self.gait_data_loader.t_step
        self.last_impact_time = 0
        self.setup_indices()
        self.setup_state_machine()
        self.set_initial_state()
        self.setup_qp()

    def setup_indices(self):
        # Indices for q and dq can remain directly assigned as before
        self.q_i = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 14, 15, 16, 17, 18, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 50, 55, 56, 57, 58, 59, 60]
        self.dq_i = [0,  1,  2,  3,  4,  5,  6,  7,  8, 12, 13, 14, 15, 16, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 44, 48, 49, 50, 51, 52, 53]

    def setup_state_machine(self):
        states = [state.name for state in ControllerStance]
        transitions = [
            {'trigger': 'switch_stance', 'source': ControllerStance.RIGHT_STANCE.name, 'dest': ControllerStance.LEFT_STANCE.name},
            {'trigger': 'switch_stance', 'source': ControllerStance.LEFT_STANCE.name, 'dest': ControllerStance.RIGHT_STANCE.name}
        ]
        self.machine = Machine(model=self, states=states, transitions=transitions, initial=ControllerStance.RIGHT_STANCE.name)

    def set_initial_state(self):
        self.data.qpos = self.model.key_qpos
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)

    def setup_qp(self):
        com_height = self.data.subtree_com[0, 2]
        self.rightStanceQP = PWQP(ControllerStance.RIGHT_STANCE, com_height)
        # self.leftStanceQP = PWQP(ControllerStance.LEFT_STANCE, com_height)
        self.ctrl = np.zeros(self.model.nu)

    def solve_qp(self):
        self.s = np.min([(self.data.time-self.last_impact_time)/self.t_step,1.0])
        # self.s = 0.0
        q, dq, _ = self.getReducedState()
                
        try:
            if self.state == ControllerStance.RIGHT_STANCE.name:
                self.rightStanceQP.Dcf.set_state(q, dq)
                q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
                # q_actuated_des = self.rightStanceQP.Dcf.get_actuated_q()
                # q_pos = self.data.qpos[self.q_i]
                # q_actuated_des = q_pos[self.q_actuated]
                # dq_actuated_des = np.zeros_like(q_actuated_des)
                # ddq_actuated_des = np.zeros_like(q_actuated_des)
                self.rightStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
                self.ctrl = self.rightStanceQP.WalkingQP()
            # elif self.state == ControllerStance.LEFT_STANCE.name:
            #     self.leftStanceQP.Dcf.set_state(q, dq)
            #     q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
            #     self.leftStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
            #     self.ctrl = self.leftStanceQP.WalkingQP()
            print('QP success')
        except:
            print('QP failed')
            
        return self.ctrl
    
    def getReducedState(self):
        q, v = self.data.qpos[self.q_i], self.data.qvel[self.dq_i]
        return q.copy(), v.copy(), self.data.time
    
    def get_desired_actuated_configuration(self, phase):
        if self.state == ControllerStance.RIGHT_STANCE.name:
            return self.gait_data_loader.evaluate_bezier_curve(phase)
        else:
            return self.gait_data_loader.evaluate_bezier_curve_relabelled(phase)

    def foot_switching_algo(self):
        q, _, _ = self.getReducedState()
        if self.state == ControllerStance.LEFT_STANCE.name:
            swingFootSpringDefection = self.leftStanceQP.SwFspring(q)
            T_SwFoot, _ = self.leftStanceQP.Dcf.get_pose_and_vel_for_site('right-foot')
        elif self.state == ControllerStance.RIGHT_STANCE.name:
            swingFootSpringDefection = self.rightStanceQP.SwFspring(q)
            T_SwFoot, _ = self.rightStanceQP.Dcf.get_pose_and_vel_for_site('left-foot')
        PSwFoot = T_SwFoot[:3, 3].reshape((3, 1))
        if self.s >= 0.5 and (PSwFoot[2] < 1e-2 or swingFootSpringDefection > 0.01):
            self.switch_stance()
            self.s = 0
            self.last_impact_time = self.data.time

    def run(self):
        while self.viewer.is_running():
            step_start = time.time()
            
            self.data.ctrl = self.solve_qp()
            mujoco.mj_step(self.model, self.data, nstep=15)
            # self.foot_switching_algo()
                # self.unpause = True

            self.sync_viewer(step_start)

    def sync_viewer(self, step_start):
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
        self.viewer.sync()
        self.maintain_realtime(step_start)

    def maintain_realtime(self, step_start):
        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

if __name__ == "__main__":
    digitSimulation = DigitSimulation()
    digitSimulation.run()
