from enum import Enum, auto
import csv

import numpy as np
import cvxpy as cp
from pyquaternion import quaternion

from CasDynamics import DigitCasadiWrapper
 
class ControllerStance(Enum):
    LEFT_STANCE = auto()
    RIGHT_STANCE = auto()
    
class PWQP():
    def __init__(self, stance: ControllerStance = ControllerStance.RIGHT_STANCE, doubleStance: bool = True):
        self.stance = stance
        self.des_com_pos = np.array([0.05, 0.0, 0.8])
        self.des_com_vel = np.array([0.0, 0.0, 0.0])
        self.Dcf = DigitCasadiWrapper()
        self.doubleStance = doubleStance
        self.totM = 47.925414
        
        self.csvfile = open('output.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)
        
        self.n_ddq = 36
        self.n_u = 20
        self.n_JacCL = 6
        self.n_JacStF = 6
        self.n_springs = 4
        
        #contact constants
        self.min_Fz = 20 #minimum up GRF
        self.mu = 0.5 #friction coeff
        self.l_foot = 0.15
        self.w_foot = 0.07
        
        self.pH = cp.Parameter((self.n_ddq, self.n_ddq), name='H')
        self.vddq = cp.Variable((self.n_ddq, 1), name='ddq')
        self.pC_terms = cp.Parameter((self.n_ddq, 1), name='C_terms')
        self.ptau = cp.Parameter((self.n_ddq, 1), name='tau')
        self.pJacCL = cp.Parameter((self.n_JacCL, self.n_ddq), name='JacCL')
        self.vlambdaCL = cp.Variable((self.n_JacCL, 1), name='lambdaCL')
        self.pJacStF = cp.Parameter((self.n_JacStF, self.n_ddq), name='JacStF')
        self.vlambdaStF = cp.Variable((self.n_JacStF, 1), name='lambdaStF')
        self.JacFsp= np.zeros((self.n_springs,self.n_ddq))
        self.JacFsp[[0,1,2,3],[10,12,25,27]]=1
        self.vlambdaFsp = cp.Variable((self.n_springs, 1), name='lambdaFsp')
        self.vu = cp.Variable((self.n_u,1),'u')
        self.pdJacCLdq = cp.Parameter((self.n_JacCL, 1), name='dJacCLdq')
        self.pdJacStFdq = cp.Parameter((self.n_JacStF, 1), name='dJacStFdq')
        self.R2StF = cp.Parameter((2,2),'R2x2StF') #stance foot rotation matrix 2x2
        self.R2SwF = cp.Parameter((2,2),'R2x2SwF') #stance foot rotation matrix 2x2
        if self.doubleStance:
            self.vlambdaSwF = cp.Variable((self.n_JacStF, 1), name='lambdaSwF')
        self.pdesddh = cp.Parameter((12,1),'desddh') #desired output space acc
        
        self.pJacSwF = cp.Parameter((self.n_JacStF,self.n_ddq),'JacSwF') #swing foot constraint
        self.pdJacSwFdq = cp.Parameter((self.n_JacStF,1),'dJacSwFdq')
        
        self.pAG = cp.Parameter((6, self.n_ddq), name='AG')
        self.pdAGdq = cp.Parameter((6, 1), name='dAGdq')
        self.pdesdhG = cp.Parameter((3, 1), name='desdhG')

        self.B = self.Dcf.get_B_matrix()
        self.vu_limit = self.Dcf.get_u_limit().reshape((self.n_u,1))
        
        self.q_act_idx, self.dq_act_idx = self.Dcf.get_actuated_indices()
        
        if not self.doubleStance:
            self.exclude_list = [4, 5, 14, 15] #exclude toe joints
        else:
            self.exclude_list = []
            
        self.q_act_idx = self.exclude_swing_foot_indices(self.q_act_idx, self.exclude_list)
        self.dq_act_idx = self.exclude_swing_foot_indices(self.dq_act_idx, self.exclude_list)
        self.qIindices = [i for i in range(20) if i not in self.exclude_list]
        
        self.pdesddq = cp.Parameter((self.n_u-len(self.exclude_list),1),'des_ddq')
        self.q_actuated_des, self.dq_actuated_des, self.ddq_actuated_des = np.zeros((self.n_u,1)), np.zeros((self.n_u,1)), np.zeros((self.n_u,1))
        
        self.WalkingProb = self.getWalkProb()
        
    def exclude_swing_foot_indices(self, arr, exclude_indices):
        # Use a list comprehension to include elements whose indices are not in exclude_indices
        return [arr[k] for k in range(len(arr)) if k not in exclude_indices]
    
    def getWalkProb(self):
        """
        Creates a cvxpy problem for Standing QP (Quadratic Programming).
        This involves assembling constraints and objectives for the optimization problem.
        """

        # Initialize the list of constraints
        Constraints = []
        Constraints.extend(self.getSingleStanceDynamics())
        Constraints.extend(self.getTorqueBoundConstraints())
        Constraints.extend(self.getStanceContactWrenchConstraint(self.vlambdaStF, self.R2StF))
        if self.doubleStance:
            Constraints.extend(self.getStanceContactWrenchConstraint(self.vlambdaSwF, self.R2SwF))

        Objective = self.getQPCost()

        # Create the optimization problem with the specified objective and constraints
        prob = cp.Problem(Objective, Constraints)

        # Ensure the problem is Disciplined Convex Programming (DCP) compliant, with Decision Problem Presolving (dpp) enabled
        assert prob.is_dcp(dpp=True), "The optimization problem is not DCP compliant with DPP enabled."

        return prob
    
    def getSingleStanceDynamics(self):
        if self.doubleStance:
            constraints = [self.pH@self.vddq + self.pC_terms == self.ptau + self.pJacCL.T@self.vlambdaCL + self.pJacStF.T@self.vlambdaStF + self.pJacSwF.T@self.vlambdaSwF + self.JacFsp.T@self.vlambdaFsp  + self.B@self.vu,  #EOM
                self.pJacCL@self.vddq + self.pdJacCLdq == 0, # closed loop constraint
                self.pJacStF@self.vddq + self.pdJacStFdq == 0,# stance foot constraint
                self.pJacSwF@self.vddq + self.pdJacSwFdq == 0,
                self.JacFsp@self.vddq ==0] #stiff spring
        else :
            constraints = [self.pH@self.vddq + self.pC_terms == self.ptau + self.pJacCL.T@self.vlambdaCL + self.pJacStF.T@self.vlambdaStF + self.JacFsp.T@self.vlambdaFsp  + self.B@self.vu,  #EOM
                self.pJacCL@self.vddq + self.pdJacCLdq == 0, # closed loop constraint
                self.pJacStF@self.vddq + self.pdJacStFdq == 0,# stance foot constraint
                self.JacFsp@self.vddq ==0] #stiff spring
        return constraints
    
    def getTorqueBoundConstraints(self):
        return [self.vu <= self.vu_limit, self.vu >= -self.vu_limit]
    
    def getStanceContactWrenchConstraint(self, vlambdaF, R):
        return [vlambdaF[5] >= self.min_Fz,                           #vertical GRF
                R@vlambdaF[[3,4]] <= (self.mu)*vlambdaF[5],     #upper friction
                R@vlambdaF[[3,4]] >= -(self.mu)*vlambdaF[5],    #lower friction
                R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1] <= (self.w_foot/2)*vlambdaF[5],            #mx
                R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1] >= -(self.w_foot/2)*vlambdaF[5],
                R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1] <= (self.l_foot/2)*vlambdaF[5],            #my
                R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1] >= -(self.l_foot/2)*vlambdaF[5],
                vlambdaF[2] >= -self.mu*(self.l_foot+self.w_foot)/2*vlambdaF[5]
                                + cp.abs(self.w_foot/2*(R[0,0]*vlambdaF[3] + R[0,1]*vlambdaF[4])-self.mu*(R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1]))
                                + cp.abs(self.l_foot/2*(R[1,0]*vlambdaF[3] + R[1,1]*vlambdaF[4])-self.mu*(R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1])),
                vlambdaF[2] <= +self.mu*(self.l_foot+self.w_foot)/2*vlambdaF[5]
                                - cp.abs(self.w_foot/2*(R[0,0]*vlambdaF[3] + R[0,1]*vlambdaF[4])+self.mu*(R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1]))
                                - cp.abs(self.l_foot/2*(R[1,0]*vlambdaF[3] + R[1,1]*vlambdaF[4])+self.mu*(R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1]))]

    def getQPCost(self):
        """
        Calculate the walking cost based on the robot's current state, desired state,
        and control inputs. The cost includes contributions from the centroidal momentum,
        specific configuration adjustments, and control efforts.

        Returns:
            objective: A cvxpy Minimize expression representing the total cost.
        """
        # Calculate the centroidal momentum cost component
        centroidal_momentum_cost = cp.sum_squares(self.pAG[0:3, :] @ self.vddq + self.pdAGdq[0:3] - self.pdesdhG)
        
        # Calculate the configuration cost component for actuated joints
        if self.stance == ControllerStance.RIGHT_STANCE:
            w2 = np.diag([10, 10, 20, 60, # left hip
                          5, 5, 5, 5, # left hand
                          60, 60, 10, 5, # right hip
                          5, 5, 5, 5]) # right hand
        elif self.stance == ControllerStance.LEFT_STANCE:
            w2 = np.diag([60, 60, 10, 5, # left hip
                          5, 5, 5, 5, # left hand
                          10, 10, 20, 60, # right hip
                          20, 20, 20, 20]) # right hand
        if self.doubleStance:
            w2 = np.diag([10, 10, 20, 60, # left hip
                          1, 1,
                          5, 5, 5, 5, # left hand
                          60, 60, 10, 5, # right hip
                          1, 1,
                          20, 20, 20, 20])
        configuration_cost = cp.sum_squares(w2 @ (self.vddq[self.dq_act_idx] - self.pdesddq))
        
        if self.doubleStance:
            # Define the weight matrix for task output deviation
            W1 = 1000.0*np.diag([1,1,1,
                            1,1,1]) # base orientation roll, pitch, com acceleration z, swing foot orientation roll, pitch

            # Calculate the task output deviation cost component
            term_A = self._getTaskOutput_ddh()[[0, 1, 2, 3, 4, 5]]
            term_B = self.pdesddh[[0, 1, 2, 3, 4, 5]]
        else:
            # Define the weight matrix for task output deviation
            W1 = np.diag([100,10,
                        10, #  base orientation x, y, z
                        #   10,
                        10,
                        100 #  com position x, y, z
                        ,100,100
                        ])
                        #   ,10]) # swing foot orientation roll, pitch, yaw

            # Calculate the task output deviation cost component
            term_A = self._getTaskOutput_ddh()[[0, 1,
                                                2, 
                                                # 3, 
                                                4,
                                                5
                                                ,6, 7
                                                ]]
                                                # , 8]]
            term_B = self.pdesddh[[0, 1,
                                2, 
                                #    3, 
                                4, 
                                5
                                ,6, 7
                                ]]
                                #    , 8]]
        
        task_output_deviation_cost = cp.sum_squares( W1 @ (term_A - term_B))

        # Calculate the control effort cost component
        control_effort_cost = cp.sum_squares(self.vu)

        # Combine all cost components into the total objective
        total_cost = (task_output_deviation_cost +
                    1 * control_effort_cost +
                    0 * centroidal_momentum_cost +
                    1*configuration_cost)

        objective = cp.Minimize(total_cost)

        return objective
    
    def WalkingQP(self):
        q, dq = self.Dcf.get_state()
        
        T_SwF, V_SwF, T_StF, _ = self.extractAndSetParameters()
        
        if self.doubleStance:
            kp = np.diag([20, 40, 20, 30, # left hip
                          20, 20,
                            10, 10, 10, 10, # left hand
                            20, 40, 20, 30, # right hip
                            20, 20,
                            10, 10, 10, 10]) # right hand
        else:
            kp = np.diag([20, 40, 20, 30, # left hip
                        10, 10, 10, 10, # left hand
                        20, 40, 20, 30, # right hip
                        10, 10, 10, 10]) # right hand
        kd = 2*np.sqrt(kp)
        
        self.pdesddq.value = (self.ddq_actuated_des[self.qIindices] - kp @ (q[self.q_act_idx] - self.q_actuated_des[self.qIindices]) - kd @ (dq[self.dq_act_idx] - self.dq_actuated_des[self.qIindices])).reshape((self.n_u-len(self.exclude_list),1))
        
        yawStF = np.arctan2(T_StF[1,0],T_StF[0,0])
        self.R2StF.value = np.array([[np.cos(yawStF),-np.sin(yawStF)],[np.sin(yawStF),np.cos(yawStF)]]).T
        
        yawSwF = np.arctan2(T_SwF[1,0],T_SwF[0,0])
        self.R2SwF.value = np.array([[np.cos(yawSwF),-np.sin(yawSwF)],[np.sin(yawSwF),np.cos(yawSwF)]]).T
        
        # Calculate desired base orientation and swing foot orientation errors
        Oebase, dOebase, OeSwF, dOeSwF = self.calculateOrientationErrors(T_SwF, V_SwF, q, dq)
        
        # Compute desired dynamics (shape, velocity, and acceleration) errors
        e, de, hdd_desired = self.computeDynamicsErrors(Oebase, dOebase, OeSwF, dOeSwF)

        # Apply control law to adjust for errors
        self._applyControlLaw(e, de, hdd_desired)

        # Solve the Walking QP
        self.WalkingProb.solve(warm_start=True, solver=cp.ECOS, verbose=False)
        # print(f"optimal value with ECOS: {self.WalkingProb.value}")
        
        # return self.pdesddq.value.squeeze()
        return self.vu.value.squeeze()
    
    def calculateOrientationErrors(self, T_SwF, V_SwF, q, dq):
        """
        Calculates orientation errors for the base and swing foot.

        Parameters:
        - T_SwF: Transformation matrix of the swing foot.
        - V_SwF: Velocity of the swing foot.
        - q: Current configuration of the robot.
        - dq: Current velocity configuration of the robot.

        Returns:
        - Oebase: Orientation error for the base.
        - dOebase: Derivative of the base orientation error.
        - OeSwF: Orientation error for the swing foot.
        - dOeSwF: Derivative of the swing foot orientation error.
        """
        # Calculate the desired base orientation from stance and swing foot positions
        quat_desbase = quaternion.Quaternion(axis=[0, 0, 1], angle=0.0)
        
        # Calculate the current base orientation and orientation error
        R_base = quaternion.Quaternion(np.array(q[3:7])).rotation_matrix
        Oebase = R_base.T @ self.QuatOriError(quat_desbase.q, np.array(q[3:7]))

        # Calculate the desired angular velocity for the base
        angvel_desbase = np.array([0, 0, 0.0])
        dOebase = dq[3:6] - 0 * angvel_desbase  # Assuming no desired change in angular velocity

        # Calculate the orientation error for the swing foot
        desired_swf_orientation = self.zero_roll_pitch(T_SwF[0:3, 0:3])
        quat_desSwF = quaternion.Quaternion(matrix=desired_swf_orientation)
        # quat_desSwF = quaternion.Quaternion(axis=[0, 0, 1], angle=0.0)              #comment
        quat_SwF = quaternion.Quaternion(matrix=T_SwF[0:3, 0:3])
        OeSwF = self.QuatOriError(quat_desSwF.q, quat_SwF.q)
        # OeSwF = np.array([0, 0, 0]) #comment
        
        # Calculate the derivative of the swing foot orientation error
        dOeSwF = V_SwF[0:3] - np.array([[0, 0, 0]]).T

        return Oebase, dOebase, OeSwF, dOeSwF
    
    def _getTaskOutput_ddh(self):
        # Calculate the derivative of the center of mass position in the XY plane
        # First, compute common terms to avoid repetition
        pAG_slice_xy = self.pAG[3:5,:] / self.totM
        pdAGdq_slice_xy = self.pdAGdq[3:5,:] / self.totM

        # Compute acceleration of the center of mass in the XY plane
        ddcomxy = pAG_slice_xy @ self.vddq + pdAGdq_slice_xy

        # Calculate the derivative of the center of mass position in the Z axis
        pAG_slice_z = self.pAG[5,:] / self.totM
        pdAGdq_slice_z = self.pdAGdq[5,:] / self.totM

        # Compute acceleration of the center of mass in the Z axis
        ddcomz = pAG_slice_z @ self.vddq + pdAGdq_slice_z

        # Assemble the final matrix h, including:
        # - Fixed base orientation acceleration (vddq[3:6])
        # - Center of mass acceleration in XY (ddcomxy)
        # - Center of mass acceleration in Z (ddcomz), reshaped to fit matrix dimensions
        # - Acceleration due to the swing foot jacobian
        h = cp.vstack((
            self.vddq[3:6], # Fixed base orientation acceleration
            ddcomxy, # Center of mass acceleration in XY
            cp.reshape(ddcomz, (1, 1)), # Center of mass acceleration in Z
            self.pJacSwF @ self.vddq + self.pdJacSwFdq # Swing foot jacobian acceleration
        ))

        return h
    
    def computeDynamicsErrors(self, Oebase, dOebase, OeSwF, dOeSwF):
        """
        Computes the dynamics errors for the control system, including shape (position) and velocity errors.

        Parameters:
        - s: Current step phase (0 to 1).
        - PSwFk_1: Previous swing foot position.
        - PSwFk: Current swing foot position.
        - T: Time duration of the step.
        - des_comz: Desired center of mass height.
        - des_x_LIP: Desired Linear Inverted Pendulum (LIP) model position.
        - Oebase, dOebase: Orientation and its derivative error for the base.
        - OeSwF, dOeSwF: Orientation and its derivative error for the swing foot.
        - Pcom, Vcom: Position and velocity of the center of mass.
        
        Returns:
        - e: Shape (position) error vector.
        - de: Velocity error vector.
        """
        # Desired shape and velocity vectors
        desired_shape_vector, desired_velocity_vector, hdd_desired =  np.zeros((12, 1)), np.zeros((12, 1)), np.zeros((12, 1))
        
        Pcom, Vcom = self.Dcf.get_pcom_vcom()
        
        PSwF = np.zeros((3,1))
        VSwF = np.zeros((3,1))
        
        # Constructing the current state vectors for position and velocity
        current_shape_vector = np.block([
            [Oebase.reshape((3, 1))],  # Current base orientation error
            [-self.des_com_pos.reshape((3,1)) + Pcom],  # Current COM position error
            [OeSwF.reshape((3, 1))],  # Current swing foot orientation error
            [PSwF.reshape((3, 1))]  # Current swing foot position error
        ])

        current_velocity_vector = np.block([
            [dOebase.reshape((3, 1))],  # Current base orientation velocity
            [-self.des_com_vel.reshape((3,1)) + Vcom],  # Current COM velocity
            [dOeSwF.reshape((3, 1))],  # Current swing foot orientation velocity
            [VSwF.reshape((3, 1))]  # Current swing foot velocity
        ])

        # Calculate shape (position) and velocity errors
        e = desired_shape_vector - current_shape_vector
        de = desired_velocity_vector - current_velocity_vector
                
        return e, de, hdd_desired
    
    def _applyControlLaw(self, e, de, hdd_desired):
        """
        Applies the control law using proportional and derivative gains to adjust for errors.

        Parameters:
        - e: Shape (position) error vector.
        - de: Velocity error vector.
        """
        # Proportional and derivative gains, defined or calculated elsewhere
        # Here, they are assumed to be class attributes or have been set prior to this method call
        kd = np.diag([20,20,10,
                      20,20,20,
                      100,100,100,
                      1,1,1])
        kp = (kd/2)**2
        
        # Compute control inputs based on PD control law
        # The control inputs are the desired accelerations to correct position and velocity errors
        self.pdesddh.value = hdd_desired.reshape((12,1)) + kp @ e.reshape((12,1)) + kd @ de.reshape((12,1))
        
        self.csvwriter.writerow(e.squeeze())
    
    def extractAndSetParameters(self):
        """Extracts parameters from the model and sets Jacobians based on the swing foot."""
        # Extract various parameters and Jacobians from the dynamics computation framework (DCF)
        self.pH.value = self.Dcf.get_H_matrix()
        self.pC_terms.value = self.Dcf.get_C_terms()
        self.ptau.value = self.Dcf.get_tau()
        self.pJacCL.value = self.Dcf.get_CLJacg()
        self.pdJacCLdq.value = self.Dcf.get_CLdJacgdq()
        self.pAG.value = self.Dcf.get_CMM()
        self.pdAGdq.value = self.Dcf.get_dCMMdq()
        
        hG_ang = self.Dcf.get_CM()[0:3]
        self.pdesdhG.value = -1*hG_ang
        
        if self.stance == ControllerStance.RIGHT_STANCE:
            stance_foot_site = 'right-foot'
            swing_foot_site = 'left-foot'
        elif self.stance == ControllerStance.LEFT_STANCE:
            stance_foot_site = 'left-foot'
            swing_foot_site = 'right-foot'
        else:
            raise ValueError('SwFoot should be -1 (left foot) or 1 (right foot)')

        # Get Jacobians for stance and swing foot
        jacStF, djacdqStF = self.Dcf.get_jac_and_djac_for_site(stance_foot_site)
        jacSwF, djacdqSwF = self.Dcf.get_jac_and_djac_for_site(swing_foot_site)
        self.pJacStF.value = jacStF
        self.pdJacStFdq.value = djacdqStF
        self.pJacSwF.value = jacSwF
        self.pdJacSwFdq.value = djacdqSwF

        # Get pose and velocity for swing
        T_SwF, V_SwF = self.Dcf.get_pose_and_vel_for_site(swing_foot_site)
        T_StF, V_StF = self.Dcf.get_pose_and_vel_for_site(stance_foot_site)

        return T_SwF, V_SwF, T_StF, V_StF
    
    def set_desired_arm_q(self, q_actuated_des, dq_actuated_des, ddq_actuated_des, des_com_pos = np.array([0.05, 0.0, 0.8]), des_com_vel = np.array([0.0, 0.0, 0.0])):
        self.q_actuated_des = q_actuated_des
        self.dq_actuated_des = dq_actuated_des
        self.ddq_actuated_des = ddq_actuated_des
        self.des_com_pos = des_com_pos
        self.des_com_vel = des_com_vel
    
    def QuatOriError(self, desQuat, curQuat):
        q_d = quaternion.Quaternion(desQuat)
        R_d = q_d.rotation_matrix
        q_c = quaternion.Quaternion(curQuat)
        R_c = q_c.rotation_matrix
        R_e = 0.5*(R_c.T@R_d - R_d.T@R_c)
        return -R_c@np.array([R_e[2,1],R_e[0,2],R_e[1,0]])
    
    def zero_roll_pitch(self, rotation_matrix):
        """
        Zeroes the roll and pitch of a given rotation matrix, keeping the yaw unchanged.
        
        Args:
        - rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
        
        Returns:
        - numpy.ndarray: A new 3x3 rotation matrix with roll and pitch zeroed, keeping yaw unchanged.
        """
        # Ensure the input is a numpy array
        rotation_matrix = np.array(rotation_matrix)
        
        # Calculate yaw using the arctan2 function to handle the correct quadrant of the angle
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Construct a new rotation matrix that only includes the yaw component
        # This matrix represents a rotation around the Y-axis (up vector)
        yaw_matrix = np.array([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)]
        ])
        
        return yaw_matrix
    
    def getActuateState(self, q, dq, q_des, dq_des):
        return q[self.q_act_idx], dq[self.dq_act_idx], q_des[self.qIindices], dq_des[self.qIindices]
    
    def SwFspring(self, q):
        if self.stance == ControllerStance.RIGHT_STANCE:
            shinSpring = -q[11]
            heelSpring = -q[13]
        else: #right swing
            shinSpring = q[26]
            heelSpring = q[28]

        return np.min([shinSpring,heelSpring])
    
    def __del__(self) -> None:
        self.csvfile.close()