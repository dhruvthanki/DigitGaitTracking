import time
import math
import copy

import numpy as np
import mujoco
import mujoco.viewer
from transitions import Machine

from cWBQP2 import PWQP, ControllerStance
from GaitDataLoader import GaitDataLoader

class DigitSimulation:
    MODEL_FILE = 'models/digit-v3.xml'
    DATA_FILE = 'Digit-data_12699.mat'
    DOUBLE_STANCE = False
    
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_FILE)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 3
        self.viewer.cam.azimuth = -120
        self.viewer.cam.elevation = -30
        self.viewer.cam.lookat = [0.0, 0.0, 0.7]
        self.gait_data_loader = GaitDataLoader(self.DATA_FILE)
        self.stored_desired = []
        self.stored_time = []
        self.stored_data = []
        self.stored_ctrl = np.zeros(self.model.nu)
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
        # if self.DOUBLE_STANCE:
        #     self.data.qpos = self.model.key_qpos[0, :]
        # else:
        self.data.qpos = self.model.key_qpos
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qvel[0] = 1.0
        mujoco.mj_forward(self.model, self.data)

    def setup_qp(self):
        self.rightStanceQP = PWQP(ControllerStance.RIGHT_STANCE, self.DOUBLE_STANCE)
        self.rightStanceQP.Dcf.set_u_limit(self.model.actuator_ctrlrange[:, 1].T)
        
        self.leftStanceQP = PWQP(ControllerStance.LEFT_STANCE, self.DOUBLE_STANCE)
        self.leftStanceQP.Dcf.set_u_limit(self.model.actuator_ctrlrange[:, 1].T)
        
        self.ctrl = np.zeros(self.model.nu)

    def solve_qp(self):
        
        q, dq, _ = self.getReducedState()
        
        if self.s > 1.0:
            print('s:', self.s)
            
        try:
            if self.state == ControllerStance.RIGHT_STANCE.name:
                self.rightStanceQP.Dcf.set_state(q, dq)
                if self.DOUBLE_STANCE:
                    q_actuated_des = self.rightStanceQP.Dcf.get_actuated_q()
                    dq_actuated_des = np.zeros_like(q_actuated_des)
                    ddq_actuated_des = np.zeros_like(q_actuated_des)
                    des_com_pos = np.array([0.05, 0.0, 0.9])
                    des_com_vel = np.array([0.0, 0.0, 0.0])
                else:
                    q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(1.0)
                    q_actuated_des[6:10] = np.array([-0.106145, 0.894838, -0.00278591, 0.344714])
                    q_actuated_des[16:20] = np.array([0.106047, -0.894876, 0.00300412, -0.344657])
                    dq_actuated_des[6:10] = np.array([0.0, 0.0, 0.0, 0.0])
                    dq_actuated_des[16:20] = np.array([0.0, 0.0, 0.0, 0.0])
                    ddq_actuated_des[6:10] = np.array([0.0, 0.0, 0.0, 0.0])
                    ddq_actuated_des[16:20] = np.array([0.0, 0.0, 0.0, 0.0])
                    des_com_pos, des_com_vel = self.gait_data_loader.getCOMTrajectory(self.s)
                # if self.s <= 1.0 and self.s >= 0.0:
                #     self.stored_desired.append(q_actuated_des)
                #     self.stored_time.append(self.data.time)
                #     self.stored_ctrl = np.vstack((self.stored_ctrl, self.ctrl))
                #     self.stored_data.append(copy.deepcopy(self.data))
                self.rightStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des, des_com_pos, des_com_vel)
                localCtrl = self.rightStanceQP.WalkingQP()
                localCtrl = np.insert(localCtrl, 4, [0.0, 0.0])
                localCtrl = np.insert(localCtrl, 14, [0.0, 0.0])
                self.ctrl = localCtrl
            elif self.state == ControllerStance.LEFT_STANCE.name:
                self.leftStanceQP.Dcf.set_state(q, dq)
                q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
                des_com_pos, des_com_vel = self.gait_data_loader.getCOMTrajectory(self.s)
                self.leftStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des, des_com_pos, des_com_vel)
                self.ctrl = self.leftStanceQP.WalkingQP()
            # if self.state == ControllerStance.RIGHT_STANCE.name:
            #     self.ctrl[[4,5]] = 0.0
            # elif self.state == ControllerStance.LEFT_STANCE.name:
            #     self.ctrl[[14, 15]] = 0.0
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
            return True
        return False

    def run(self):
        while self.viewer.is_running():
            step_start = time.time()
            
            self.s = np.min([(self.data.time-self.last_impact_time)/self.t_step,1.0])
            
            if self.s < 1.0:
                self.data.ctrl = self.solve_qp()
                mujoco.mj_step(self.model, self.data, nstep=15)
            # if not self.DOUBLE_STANCE:
            #     impact_detected = self.foot_switching_algo()
            # if math.isclose(self.s, 1.0, rel_tol=1e-9, abs_tol=0.0):
            #     break
                
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
    
    def close(self):
        self.viewer.close()

if __name__ == "__main__":
    import pickle
    digitSimulation = DigitSimulation()
    digitSimulation.run()
    digitSimulation.close()
    
    # data_to_store = {'mjdata': digitSimulation.stored_data, 'desired': digitSimulation.stored_desired}
    # with open('my_list.pkl', 'wb') as file:
    #     pickle.dump(data_to_store, file)
    
    # import matplotlib.pyplot as plt
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_data)
    
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,0])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,1])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,2])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,3])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,4])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,5])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,6])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,7])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,8])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,9])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,10])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,11])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,12])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,13])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,14])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,15])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,16])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,17])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,18])
    # plt.plot(digitSimulation.stored_time, digitSimulation.stored_ctrl[1:,19])
    # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
    # plt.xlim([0, 0.5])
    # plt.ylim([-0.9, 0.9])
    # plt.show()