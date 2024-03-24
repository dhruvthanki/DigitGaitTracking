import time
from enum import Enum, auto

import numpy as np
import mujoco
import mujoco.viewer
from transitions import Machine

from cWBQP import PWQP
from C_MPCC import LIPMPC
from angles import Angle

class StanceState(Enum):
    RIGHT_STANCE = auto()
    LEFT_STANCE = auto()

class DigitSimulation:
    def __init__(self, model_file='models/digit-v3.xml'):
        self.model = mujoco.MjModel.from_xml_path(model_file)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        states = [state.name for state in StanceState]
        transitions = [
            {'trigger': 'switch_stance', 'source': StanceState.RIGHT_STANCE.name, 'dest': StanceState.LEFT_STANCE.name},
            {'trigger': 'switch_stance', 'source': StanceState.LEFT_STANCE.name, 'dest': StanceState.RIGHT_STANCE.name}
        ]
        self.machine = Machine(model=self, states=states, transitions=transitions, initial=StanceState.RIGHT_STANCE.name)
        self.Tk_1 = self.t_MPC_lastrun = 0
        self.initialize_state()
        
        self.PWQP = PWQP()
        self.MPC = LIPMPC({'H': self.data.subtree_com[0, 2]})
        
        self.q_i = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 14, 15, 16, 17, 18, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 50, 55, 56, 57, 58, 59, 60]
        self.dq_i = [0,  1,  2,  3,  4,  5,  6,  7,  8, 12, 13, 14, 15, 16, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 44, 48, 49, 50, 51, 52, 53]
        
        q, dq, _ = self.output()
        self.PWQP.Dcf.set_state(q, dq)
        q_actuated = self.PWQP.Dcf.get_actuated_q()
        self.PWQP.set_desired_arm_q(q_actuated)
        
        self.update_robot_state()
        self.T_SwFoot, _ = self.select_foot()
        self.update_swing_foot_state()

        self.ctrl = np.zeros(self.model.nu)

    def initialize_state(self):
        self.data.qpos = self.get_initial_positions()
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def get_initial_positions():
        return np.array([
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
            ])

    def update_robot_state(self):
        q, dq, _ = self.output()
        self.PWQP.Dcf.set_state(q, dq)

    def update_swing_foot_state(self):
        self.thSwFk_1 = np.arctan2(self.T_SwFoot[1, 0], self.T_SwFoot[0, 0])
        self.PSwFk_1 = self.T_SwFoot[:2, 3].reshape((2, 1))

    def output(self):
        q, v = self.data.qpos[self.q_i], self.data.qvel[self.dq_i]
        return q.copy(), v.copy(), self.data.time

    def select_foot(self):
        isRightStance = self.state == StanceState.RIGHT_STANCE.name
        sw_foot_site = 'left-foot' if isRightStance else 'right-foot'
        st_foot_site = 'right-foot' if isRightStance else 'left-foot'
        T_SwFoot, _ = self.PWQP.Dcf.get_pose_and_vel_for_site(sw_foot_site)
        T_StFoot, _ = self.PWQP.Dcf.get_pose_and_vel_for_site(st_foot_site)
        return T_SwFoot, T_StFoot

    def solve_QP(self):
        q, dq, _ = self.output()
        
        self.s = np.min([(self.data.time-self.Tk_1)/self.MPC.Tst,1.0])
        
        isRightStance = self.state == StanceState.RIGHT_STANCE.name
        
        ##### Solve QP
        try:
            self.ctrl = self.PWQP.WalkingQP(q,dq,isRightStance,self.s,self.PSwFk_1,self.thSwFk_1,self.MPC.H,self.MPC.Tst,np.array([0.0,0.0,0]),np.zeros((4,1)),0.0)
        except:
            print('QP failed')
            
        return self.ctrl

    def foot_switching_algo(self):
        # Fetch current robot state
        q, _, _ = self.output()
        
        # Extract the positions of the stance and swinging feet
        PStFoot = self.T_StFoot[:3, 3].reshape((3, 1))
        PSwFoot = self.T_SwFoot[:3, 3].reshape((3, 1))

        # Determine if it's time to switch feet based on predefined criteria:
        # 1. Progress (self.s) is at least 0.5, indicating halfway through the swing phase.
        # 2. The swinging foot's height is below a small threshold (practically touching the ground)
        #    or the spring force exceeds a threshold, indicating contact.
        isRightStance = self.state == StanceState.RIGHT_STANCE.name
        if self.s >= 0.5 and (PSwFoot[2] < 1e-2 or self.PWQP.SwFspring(q, isRightStance) > 0.01):
            # Switch stance state
            self.switch_stance()
            
            # Reset timing and progress variables
            self.s = 0
            self.Tk_1 = self.t_MPC_lastrun = self.data.time
            
            # Update the position and angle of the swinging foot to become the new stance foot
            self.PSwFk_1 = PStFoot[:2]
            self.thSwFk_1 = self.update_swing_foot_angle(self.T_SwFoot)

    def update_swing_foot_angle(self, T_SwFoot):
        """Computes and unwraps the angle of the swinging foot based on its current transformation matrix."""
        angle_current = np.arctan2(T_SwFoot[1, 0], T_SwFoot[0, 0])
        return self.get_unwrapped_theta_to_last(angle_current, self.thSwFk_1)

    def run(self):
        while self.viewer.is_running():
            step_start = time.time()
            self.update_robot_state()
            self.T_SwFoot, self.T_StFoot = self.select_foot()
            ctrl = self.solve_QP()
            self.foot_switching_algo()

            # Step simulation forward
            self.data.ctrl = ctrl
            mujoco.mj_step(self.model, self.data, nstep=15)

            # Synchronize the viewer with the simulation state
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
            self.viewer.sync()

            # Maintain real-time simulation speed
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    def get_unwrapped_theta_to_last(self, theta, theta_last):
        diff_theta = (Angle(theta) - Angle(theta_last))
        return theta_last + diff_theta.toRadian

if __name__ == "__main__":
    digitSimulation = DigitSimulation()
    digitSimulation.run()
