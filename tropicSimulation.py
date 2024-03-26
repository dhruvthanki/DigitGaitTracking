import time

import keyboard
import numpy as np
import mujoco
import mujoco.viewer
from transitions import Machine

from cWBQP2 import PWQP, ControllerStance
from GaitDataLoader import GaitDataLoader

class DigitSimulation:
    MODEL_FILE = 'models/digit-v3.xml'
    DATA_FILE = 'Digit-data_12699.mat'

    def __init__(self):
        self.unpause = False
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_FILE)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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
        self.leftStanceQP = PWQP(ControllerStance.LEFT_STANCE, com_height)
        self.ctrl = np.zeros(self.model.nu)

    def solve_qp(self):
        self.s = np.min([(self.data.time-self.last_impact_time)/self.t_step,1.0])
        q, dq, _ = self.getReducedState()
                
        try:
            if self.state == ControllerStance.RIGHT_STANCE.name:
                self.rightStanceQP.Dcf.set_state(q, dq)
                q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
                self.rightStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
                self.ctrl = self.rightStanceQP.WalkingQP()
            elif self.state == ControllerStance.LEFT_STANCE.name:
                self.leftStanceQP.Dcf.set_state(q, dq)
                q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
                self.leftStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
                self.ctrl = self.leftStanceQP.WalkingQP()
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
