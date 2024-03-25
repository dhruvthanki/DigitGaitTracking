import time
import numpy as np
import mujoco
import mujoco.viewer
from transitions import Machine

from cWBQP2 import PWQP, ControllerStance
from GaitDataLoader import GaitDataLoader

class DigitSimulation:
    MODEL_FILE = 'models/digit-v3.xml'
    DATA_FILE = 'Digit-data_12699.mat'
    INITIAL_POSITIONS = np.array([
            -5.302706370333674230e-03, 3.449528810018998915e-04, 8.322345036558366216e-01,
            1, 0, 0, 0, 2.983497082849632109e-01, -4.609038906699547411e-02,
            -1.028745509393533958e-02, -6.180393852816212785e-01, 7.793554143652230426e-01,
            -8.069510787823898357e-02, 6.419311415445301539e-02, -2.070011726210992664e-01,
            -2.244121380834677071e-04, 2.110699778217571820e-01, -9.877097799413092627e-04,
            1.236112663484994628e-01, 9.951418147941888392e-01, -7.412700813079739492e-02,
            8.443159191734769115e-03, -6.423914831758105459e-02, -1.744582995717965102e-01,
            9.882915693413950597e-01, -1.212740307946869184e-01, -1.727696856720108143e-02,
            9.096092447122425262e-02, -1.432831937253021826e-01, -9.104605353826955572e-02,
            -0.106437, 0.89488, -1.860540486313262858e-05, 0.344684, -2.920205202074338535e-01,
            4.328204378325633400e-02, 1.147273422695336415e-02, 7.911839121390622509e-01,
            -6.027482048891177335e-01, -6.289769939569501978e-02, 8.225872650365767536e-02,
            2.078483653506341677e-01, 2.582094172140779226e-04, -2.120813876878341331e-01,
            1.025737127413941900e-03, -1.242014477753271978e-01, 9.949675536918867191e-01,
            7.617350960264548942e-02, 8.627290683723530182e-03, 6.451924821831753198e-02,
            1.758658078088930210e-01, 9.790791561883002148e-01, 1.808192819113194350e-01,
            -2.293005118224257163e-02, -9.045775787327837991e-02, 1.443733939669925581e-01,
            9.228187573855245462e-02, 0.106437, -0.89488, 1.860540486313262858e-05, -0.344684
            ])  # Use the given long array here
    
    INITIAL_GAIT_CONFIGURATION = np.array([
        -7.3031e-02, 7.5111e-02, 0.90999, 1, 0, 0, 0,
        0.37521, -9.94268e-05, 0.430631, 0.999564, -0.0155778, -0.000233017,
        0.025092, 0.0578272, -0.00851288, -0.0338148, -0.00505735, -0.268574,
        0.956675, 0.257459, 0.030493, 0.132508, 0.268626, 0.754038, -0.642522,
        0.0939687, -0.0987982, 0.301688, -4.18302e-05, -0.16119, 0.373506,
        -0.913527, 1.22848, -0.375273, 9.981e-05, 0.0232603, 0.999949,
        0.00884147, -8.16236e-05, -0.00481516, -0.0119976, 0.00270197,
        -0.00144796, 0.00371619, -0.152306, 0.96285, -0.257929, -0.0177366,
        0.0779648, 0.152309, 0.759705, 0.645243, -0.0553993, -0.0586549,
        0.147446, -3.0347e-05, -0.21315, -1.60889, 0.100992, -0.420158
        ])

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_FILE)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.gait_data_loader = GaitDataLoader(self.DATA_FILE)
        self.initialize_simulation()
    
    def initialize_simulation(self):
        self.t_step = self.gait_data_loader.t_step[0,0][0,0]
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
        self.data.qpos = self.INITIAL_POSITIONS
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)

    def setup_qp(self):
        com_height = self.data.subtree_com[0, 2]
        self.rightStanceQP = PWQP(ControllerStance.RIGHT_STANCE, com_height)
        # self.leftStanceQP = PWQP(ControllerStance.LEFT_STANCE, com_height)
        self.ctrl = np.zeros(self.model.nu)

    def solve_qp(self):
        # self.s = np.min([(self.data.time-self.last_impact_time)/self.t_step,1.0])
        self.s = 0.0
        q, dq, _ = self.getReducedState()
        
        # q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
        
        try:
            # if self.state == ControllerStance.LEFT_STANCE.name:
                # self.leftStanceQP.Dcf.set_state(q, dq)
                
                
                # self.leftStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
                # self.ctrl = self.leftStanceQP.WalkingQP()
            # elif self.state == ControllerStance.RIGHT_STANCE.name:
            self.rightStanceQP.Dcf.set_state(q, dq)
            q_actuated_des = self.rightStanceQP.Dcf.get_actuated_q()
            dq_actuated_des = np.zeros_like(q_actuated_des)
            ddq_actuated_des = np.zeros_like(q_actuated_des)
            self.rightStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
            self.ctrl = self.rightStanceQP.WalkingQP()
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
