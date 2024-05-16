import time
import math

from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco
import mujoco.viewer
from transitions import Machine

from cWBQP2 import PWQP, ControllerStance
from GaitDataLoader import GaitDataLoader, q_i, q_actuated
from DataLogger import RobotDataLogger

is_paused = True
sim_toggle = time.time()
import keyboard
def toggle_pause():
    global is_paused, sim_toggle
    sim_toggle = time.time()
    is_paused = not is_paused
    if is_paused:
        print("Simulation paused.")
    else:
        print("Simulation resumed.")
keyboard.add_hotkey('space', toggle_pause)
 
class DigitSimulation:
    MODEL_FILE = 'models/digit-v3.xml'
    DATA_FILE = 'Digit-data_12699.mat'
    DOUBLE_STANCE = False
    
    def __init__(self):
        self.impact_detected = False
        self.logger = RobotDataLogger()
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_FILE)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 5
        self.viewer.cam.azimuth = -90
        self.viewer.cam.elevation = -30
        self.viewer.cam.lookat = [0.0, 0.0, 0.7]
        self.gait_data_loader = GaitDataLoader(self.DATA_FILE)
        self.contant_force = 0.0
        self.randon_basis = np.ones(6)
        self.setOnce = True
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
        
        try:
            if self.state == ControllerStance.RIGHT_STANCE.name:
                self.rightStanceQP.Dcf.set_state(q, dq)
                if self.DOUBLE_STANCE and self.setOnce:
                    q_actuated_des = self.rightStanceQP.Dcf.get_actuated_q()
                    dq_actuated_des = np.zeros_like(q_actuated_des)
                    ddq_actuated_des = np.zeros_like(q_actuated_des)
                    # self.setOnce = False
                else:
                    q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
                    q_actuated_des[6:10] = np.array([-0.106145, 0.894838, -0.00278591, 0.344714])
                    q_actuated_des[16:20] = np.array([0.106047, -0.894876, 0.00300412, -0.344657])
                    dq_actuated_des[6:10] = np.array([0.0, 0.0, 0.0, 0.0])
                    dq_actuated_des[16:20] = np.array([0.0, 0.0, 0.0, 0.0])
                    ddq_actuated_des[6:10] = np.array([0.0, 0.0, 0.0, 0.0])
                    ddq_actuated_des[16:20] = np.array([0.0, 0.0, 0.0, 0.0])
                self.rightStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
                self.ctrl = self.rightStanceQP.WalkingQP()
                
            elif self.state == ControllerStance.LEFT_STANCE.name:
                self.leftStanceQP.Dcf.set_state(q, dq)
                q_actuated_des, dq_actuated_des, ddq_actuated_des = self.get_desired_actuated_configuration(self.s)
                q_actuated_des[6:10] = np.array([-0.106145, 0.894838, -0.00278591, 0.344714])
                q_actuated_des[16:20] = np.array([0.106047, -0.894876, 0.00300412, -0.344657])
                dq_actuated_des[6:10] = np.array([0.0, 0.0, 0.0, 0.0])
                dq_actuated_des[16:20] = np.array([0.0, 0.0, 0.0, 0.0])
                ddq_actuated_des[6:10] = np.array([0.0, 0.0, 0.0, 0.0])
                ddq_actuated_des[16:20] = np.array([0.0, 0.0, 0.0, 0.0])
                self.leftStanceQP.set_desired_arm_q(q_actuated_des, dq_actuated_des, ddq_actuated_des)
                self.ctrl = self.leftStanceQP.WalkingQP()
            
            if self.state == ControllerStance.RIGHT_STANCE.name:
                siteID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'left-foot')
            else:
                siteID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'right-foot')
            
            torso_rpy = self.quaternion_to_rpy(self.data.qpos[3:7])
            swingFoot_ori = R.from_matrix(self.data.site_xmat[siteID, :].reshape((3, 3)))
            swingFoot_rpy = swingFoot_ori.as_euler('xyz', degrees=True)
            
            torseID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
            mujoco.mj_subtreeVel(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            
            torque_limits = self.leftStanceQP.Dcf.get_u_limit().reshape((self.leftStanceQP.n_u,1))
            percent_to_max_torque = (np.abs(self.ctrl) / torque_limits.squeeze()) * 100
            
            if self.state == ControllerStance.RIGHT_STANCE.name:
                q_curr, dq_curr, q_act, dq_act = self.rightStanceQP.getActuateState(q, dq, q_actuated_des, dq_actuated_des)
            else:
                q_curr, dq_curr, q_act, dq_act = self.leftStanceQP.getActuateState(q, dq, q_actuated_des, dq_actuated_des)
                
            q_err = q_act - q_curr
            dq_err = dq_act - dq_curr
            
            cam = self.data.subtree_angmom[torseID, :]
            comPos = self.data.subtree_com[torseID,:]
            
            self.logger.log_data(self.data.time,
                                 np.rad2deg(q_err),
                                 np.rad2deg(dq_err),
                                 np.rad2deg(q_act),
                                 np.rad2deg(dq_act),
                                 percent_to_max_torque,
                                 torso_rpy,
                                 swingFoot_rpy,
                                 cam,
                                 comPos)
        except:
            print('QP failed')
        
        # self.ctrl[[4, 5, 14, 15]] = 0.0
        return self.ctrl
    
    def getReducedState(self):
        q, v = self.data.qpos[self.q_i], self.data.qvel[self.dq_i]
        return q.copy(), v.copy(), self.data.time
    
    def get_desired_actuated_configuration(self, phase):
        if self.state == ControllerStance.RIGHT_STANCE.name:
            return self.gait_data_loader.evaluate_bezier_curve(phase)
        else:
            relablling_idx = [10, 11, 12, 13, 14, 15, 
                          6, 7, 8, 9, 
                          0, 1, 2, 3, 4, 5, 
                          16, 17, 18, 19]
            q_des, dq_des, ddq_des = self.gait_data_loader.evaluate_bezier_curve(phase)
            return -q_des[relablling_idx], -dq_des[relablling_idx], -ddq_des[relablling_idx]

    def foot_switching_algo(self):
        q, _, _ = self.getReducedState()
        if self.state == ControllerStance.LEFT_STANCE.name:
            swingFootSpringDefection = self.leftStanceQP.SwFspring(q)
            T_SwFoot, _ = self.leftStanceQP.Dcf.get_pose_and_vel_for_site('right-foot')
        elif self.state == ControllerStance.RIGHT_STANCE.name:
            swingFootSpringDefection = self.rightStanceQP.SwFspring(q)
            T_SwFoot, _ = self.rightStanceQP.Dcf.get_pose_and_vel_for_site('left-foot')
        PSwFoot = T_SwFoot[:3, 3].reshape((3, 1))
        if self.s >= 0.5 and (PSwFoot[2] < 2e-2):# or swingFootSpringDefection > 0.05):
            self.switch_stance()
            print('Switching Stance at time:', self.data.time, 'State:', self.state)
            # self.data.qvel[0] += 0.5
            self.s = 0
            self.last_impact_time = self.data.time
            self.logger.saveSwitchingTimes(self.data.time)
            return True
        return False

    def run(self):
        while self.viewer.is_running():
            global is_paused, sim_toggle
            # if not is_paused and self.setOnce:
            #     sim_start = time.time()
            #     self.setOnce = False
            # else:
            #     self.setOnce = True
            step_start = time.time()
            
            self.s = np.min([(self.data.time-self.last_impact_time)/self.t_step,1.0])
            
            if self.data.time-self.contant_force>0.01: # 100Hz
                self.contant_force = self.data.time
                self.randon_basis = np.random.randn(6)
            
            if not is_paused:
                # if self.data.time > 1.47:
                #     is_paused = True
                #     print(self.s)
                self.data.ctrl = self.solve_qp()
                torseID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
                # if self.data.time > 0.7 and self.data.time < 1.0:
                #     self.data.xfrc_applied[torseID,:] = 50.0*self.randon_basis
                mujoco.mj_step(self.model, self.data, nstep=15)
                if not self.DOUBLE_STANCE:
                    self.impact_detected = self.foot_switching_algo()
            
            self.sync_viewer(step_start, self.randon_basis[:3])
            
            if (time.time() - sim_toggle > 5) and not is_paused:
                print('Simulation time:', self.data.time)
                break

    def sync_viewer(self, step_start, endPoint):
        global is_paused
        if not is_paused:
            with self.viewer.lock():
                torseID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
                base_pos = self.data.xpos[torseID,:]
                comPos = self.data.subtree_com[torseID,:]
                
                if endPoint.all()!=0.0:
                    length = np.linalg.norm(endPoint)
                
                rot = self.orientation_matrix(base_pos, base_pos + endPoint)
                
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
                # mujoco.mjv_initGeom(
                #     self.viewer.user_scn.geoms[0],
                #     type=mujoco.mjtGeom.mjGEOM_ARROW,
                #     size=[0.01, 0.01, length],
                #     pos=base_pos,
                #     mat=rot.flatten(),
                #     rgba=np.array([1,0,0,1])
                #     )
                # self.viewer.user_scn.ngeom = 1
        self.viewer.sync()
        self.maintain_realtime(step_start)

    def maintain_realtime(self, step_start):
        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    
    def close(self):
        self.logger.save_data()
        self.viewer.close()
        
    def quaternion_to_rpy(self, quaternion):
        """
        Convert a quaternion to roll, pitch, and yaw angles using SciPy's Rotation module.
        
        Parameters:
        quaternion (list or np.ndarray): Quaternion represented as [w, x, y, z].
        
        Returns:
        tuple: Roll, pitch, and yaw angles in radians.
        """
        # Create a Rotation object from the quaternion
        rotation = R.from_quat(quaternion)
        
        # Convert to Euler angles with the 'xyz' convention
        rpy = rotation.as_euler('xyz', degrees=True)
        
        # Return the roll, pitch, and yaw angles
        return rpy
    
    def orientation_matrix(self, point1, point2):
        # Calculate the vector connecting the two points
        vector = np.array(point2) - np.array(point1)
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("The two points are the same, so the vector length is zero.")
        normalized_vector = vector / norm
        
        # Create the orientation matrix
        # For simplicity, let's assume the matrix aligns the vector with the x-axis.
        # This means we will find a rotation matrix that aligns the vector to [1, 0, 0].
        
        # Calculate the axis of rotation (cross product with x-axis)
        x_axis = np.array([1, 0, 0])
        axis_of_rotation = np.cross(x_axis, normalized_vector)
        axis_norm = np.linalg.norm(axis_of_rotation)
        
        if axis_norm == 0:
            # If the axis norm is 0, the vectors are already aligned
            if np.dot(x_axis, normalized_vector) > 0:
                return np.eye(3)  # No rotation needed
            else:
                # 180 degrees rotation around any perpendicular axis, e.g., y-axis
                return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        axis_of_rotation = axis_of_rotation / axis_norm
        
        # Calculate the angle of rotation (dot product with x-axis)
        angle = np.arccos(np.dot(x_axis, normalized_vector))
        
        # Create the rotation matrix using Rodrigues' rotation formula
        K = np.array([[0, -axis_of_rotation[2], axis_of_rotation[1]],
                    [axis_of_rotation[2], 0, -axis_of_rotation[0]],
                    [-axis_of_rotation[1], axis_of_rotation[0], 0]])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R


if __name__ == "__main__":
    
    digitSimulation = DigitSimulation()
    
    digitSimulation.run()
    digitSimulation.close()
