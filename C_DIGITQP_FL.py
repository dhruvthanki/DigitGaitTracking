import numpy as np
from pyquaternion import quaternion
import cvxpy as cp
from enum import Enum, auto

from angles import Angle
from CasDynamics import DigitCasadiWrapper

class ControllerStance(Enum):
    LEFT_STANCE = auto()
    RIGHT_STANCE = auto()
    
class PWQP():
    def __init__(self, stance: ControllerStance = ControllerStance.RIGHT_STANCE):
        self.stance = stance
        self.Dcf = DigitCasadiWrapper() #digit casadi functions
        
        #constants
        n_ddq = 36
        n_u = 20
        n_lambda = 18
        n_v = n_ddq + n_u + n_lambda
        self.zCL = 0.08 #walking clearence
        self.totM = 47.925414
        self.g = 9.806
        
        # Defining variables
        self.vddq = cp.Variable((n_ddq,1),'ddq')
        self.vu = cp.Variable((n_u,1),'u')
        self.vlambdaCL = cp.Variable((6,1),'lambda_CL') #for closed loop
        self.vlambdaStF = cp.Variable((6,1),'lambda_Stf') #for stance foot
        self.vlambdaSwF = cp.Variable((6,1),'lambda_Swf') #for swing foot if considering on ground
        self.vlambdaSwFsp = cp.Variable((2,1),'lambda_F_spring') #making spring of Leg inf stiff
        self.vetaStF = cp.Variable((6,1),'etaStF') #relaxaction factor for stance foot contact
        self.vetaSwF = cp.Variable((6,1),'etaStwF') # on swing foot
        self.vlambdaFsp = cp.Variable((4,1),'lambda_Fsp') #stiffsprings
        
        #define parameters| p-heading for parameters
        self.pH = cp.Parameter((n_ddq,n_ddq),'H')
        self.pC_terms = cp.Parameter((n_ddq,1),'C_terms')
        self.ptau = cp.Parameter((n_ddq,1),'tau')

        self.pJacCL = cp.Parameter((6,n_ddq),'JacCL') #closed loop constraint
        self.pdJacCLdq = cp.Parameter((6,1),'dJacCLdq')
        
        self.R2StF = cp.Parameter((2,2),'R2x2StF') #stance foot rotation matrix 2x2
        self.pJacStF = cp.Parameter((6,n_ddq),'JacStF') #stance foot constraint
        self.pdJacStFdq = cp.Parameter((6,1),'dJacStFdq')
        self.pJacSwF = cp.Parameter((6,n_ddq),'JacSwF') #swing foot constraint
        self.pdJacSwFdq = cp.Parameter((6,1),'dJacSwFdq') 
        self.pdesddPSwF = cp.Parameter((6,1),'desddPSwF') #swing foot desired acceleration of position and orientation, to be used when in air
        self.JacFsp= np.zeros((4,n_ddq))
        self.JacFsp[[0,1,2,3],[10,12,25,27]]=1

        self.pdesddq = cp.Parameter((8,1),'des_ddq') #desired actuated acceleration

        self.pAG = cp.Parameter((6,n_ddq),'AG') #CMM
        self.pdAGdq = cp.Parameter((6,1),'dAGdq') #dCMMdq
        self.pdesdhG = cp.Parameter((6,1),'deshG') #desired centroidal momentum

        self.pdesddbase = cp.Parameter((3,1),'desddbase') #desired base acc

        self.pdesddh = cp.Parameter((12,1),'desddh') #desired output space acc
        self.pJacBase = cp.Parameter((3,n_ddq),'JacBase')
        self.pdJacBasedq = cp.Parameter((3,1),'dJacBasedq')
        
        self.B = self.Dcf.get_B_matrix() #input mapping matrix
        
        self.ctrlrange = 1*np.array([[ -1.4,  -1.4, -12.5, -12.5,  -0.9,  -0.9,  -1.4,  -1.4,  -1.4,
                                        -1.4,  -1.4,  -1.4, -12.5, -12.5,  -0.9,  -0.9,  -1.4,  -1.4,
                                        -1.4,  -1.4],
                                    [  1.4,   1.4,  12.5,  12.5,   0.9,   0.9,   1.4,   1.4,   1.4,
                                        1.4,   1.4,   1.4,  12.5,  12.5,   0.9,   0.9,   1.4,   1.4,
                                        1.4,   1.4]]).T

        #contact constants
        self.min_Fz = 20 #minimum up GRF
        self.mu = 0.5 #friction coeff
        self.l_foot = 0.15
        self.w_foot = 0.07 #for xaxis
        self.z_c = 2 #for zaxis
        
        #formulate problems,
        # walking
        self.WalkingProb = self._getWalkProb()
        self.hdesired = np.zeros((20,1))

        self.Cur_e_h= np.zeros((12,1)) #error of the custom output
        self.Cur_hang = np.zeros((3,1)) #hangular momentum
        
    "getConstraints"
    def _getEOMSSConstraint(self): # EOM of Single support base
        constraints = [self.pH@self.vddq + self.pC_terms == self.ptau + self.pJacCL.T@self.vlambdaCL + self.pJacStF.T@self.vlambdaStF + self.JacFsp.T@self.vlambdaFsp  + self.B@self.vu,  #EOM
                        self.pJacCL@self.vddq + self.pdJacCLdq == 0, # closed loop constraint
                        self.pJacStF@self.vddq + self.pdJacStFdq == 0,# stance foot constraint
                        self.JacFsp@self.vddq ==0] #stiff spring
        return constraints

    def _getFootContactConstraintWrench(self,vlambdaF,R): #same for either of foot hence parameteri Caron2015ICRA
        constraints = [vlambdaF[5] >= self.min_Fz,                           #vertical GRF
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
        return constraints

    def _getInputConstraints(self): #actuator input torque
        constraints = [self.ctrlrange[:,0].reshape(20,1)<=self.vu, #lower bound on u    
                        self.ctrlrange[:,1].reshape(20,1)>=self.vu]#upper bound on u]
        return constraints
        
    "Walking Controller"
    def _getWalkingCost(self):
        """
        Calculate the walking cost based on the robot's current state, desired state,
        and control inputs. The cost includes contributions from the centroidal momentum,
        specific configuration adjustments, and control efforts.

        Returns:
            objective: A cvxpy Minimize expression representing the total cost.
        """
        # Calculate the centroidal momentum cost component
        centroidal_momentum_cost = cp.sum_squares(self.pAG[0:3, :] @ self.vddq + self.pdAGdq[0:3] - self.pdesdhG[0:3])

        # Calculate the configuration cost component for specified joints
        configuration_cost_indices = [17, 18, 19, 20, 32, 33, 34, 35]
        desired_configuration_indices = [6, 7, 8, 9, 16, 17, 18, 19]
        configuration_cost = cp.sum_squares(self.vddq[configuration_cost_indices] - self.pdesddq)

        # Define the weight matrix for task output deviation
        W1 = np.diag([1] * 12) * np.sqrt(100)

        # Calculate the task output deviation cost component
        task_output_deviation_cost = cp.sum_squares(W1 @ (self._getTaskOutput_ddh() - self.pdesddh))

        # Calculate the control effort cost component
        control_effort_cost = cp.sum_squares(self.vu)

        # Combine all cost components into the total objective
        total_cost = (task_output_deviation_cost +
                    0.1 * control_effort_cost +
                    3 * centroidal_momentum_cost +
                    0.5 * configuration_cost)

        objective = cp.Minimize(total_cost)

        return objective

    def _getWalkProb(self):
        """
        Creates a cvxpy problem for Standing QP (Quadratic Programming).
        This involves assembling constraints and objectives for the optimization problem.
        """

        # Initialize the list of constraints
        Constraints = []

        # Add dynamics constraints based on the Equations of Motion in Single Support (SS) phase
        Constraints.extend(self._getEOMSSConstraint())

        # Add constraints on the input variables (e.g., actuator limits)
        Constraints.extend(self._getInputConstraints())

        # Add foot contact constraints to ensure the stance foot applies the correct wrench
        Constraints.extend(self._getFootContactConstraintWrench(self.vlambdaStF, self.R2StF))

        # Define the objective function for the walking optimization problem
        Objective = self._getWalkingCost()

        # Create the optimization problem with the specified objective and constraints
        prob = cp.Problem(Objective, Constraints)

        # Ensure the problem is Disciplined Convex Programming (DCP) compliant, with Decision Problem Presolving (dpp) enabled
        assert prob.is_dcp(dpp=True), "The optimization problem is not DCP compliant with DPP enabled."

        # Return the constructed optimization problem
        return prob
    
    def WalkingQP(self, q, dq, isRightStance, s, PSwFk_1, PSwFk, thSwFk_1, des_comz, T, des_x_LIP, des_thSwFk):
        """
        Formulates and solves a walking Quadratic Program (QP) for the robot.

        Parameters:
        - q: Current configuration of the robot (37x1).
        - dq: Current velocity configuration of the robot (36x1).
        - SwFoot: Indicates the swing foot ('left': -1, 'right': 1).
        - s: Current step phase (0 to 1).
        - PSwFk_1: Previous swing foot position.
        - PSwFk: Current swing foot position.
        - thSwFk_1: Previous swing foot orientation.
        - des_comz: Desired center of mass height.
        - T: Time duration of the step.
        - des_x_LIP: Desired position in the Linear Inverted Pendulum model.
        - des_thSwFk: Desired final swing foot orientation.

        Returns:
        - The solution to the QP as control inputs for the robot.
        """
        
        # Initialization
        SwFoot = -1 if isRightStance else 1
        self._initializeDesiredValues()

        # Extract parameters and set Jacobians
        self._extractAndSetParameters(SwFoot)
        
        hG_ang = self.Dcf.get_CM()[0:3]
        self.pdesdhG.value[0:3] = -1*hG_ang
        
        self.pdesddq.value = (-100*(q[[18,19,20,21, 33,34,35,36]]-self.handDesqpos()) -10*dq[[17,18,19,20, 32,33,34,35]]).reshape((8,1))
        
        # Set foot orientations and velocities based on swing foot side
        T_SwF, V_SwF, T_StF, V_StF = self._setFootOrientationsAndVelocities(SwFoot)
        
        yawStF = np.arctan2(T_StF[1,0],T_StF[0,0])
        self.R2StF.value = np.array([[np.cos(yawStF),-np.sin(yawStF)],[np.sin(yawStF),np.cos(yawStF)]]).T
        
        # Calculate desired base orientation and swing foot orientation errors
        Oebase, dOebase, OeSwF, dOeSwF = self._calculateOrientationErrors(T_SwF, V_SwF, T_StF, V_StF, thSwFk_1, des_thSwFk, q, dq, des_x_LIP, s, T)
        
        # Compute desired dynamics (shape, velocity, and acceleration) errors
        e, de, hdd_desired = self._computeDynamicsErrors(s, PSwFk_1, PSwFk, T, des_comz, des_x_LIP, Oebase, dOebase, OeSwF, dOeSwF, T_SwF, V_SwF)

        # Apply control law to adjust for errors
        self._applyControlLaw(e, de, hdd_desired)

        # Solve the Walking QP
        self.WalkingProb.solve(warm_start=True, solver=cp.ECOS, verbose=False)

        return self.vu.value.squeeze()

    def _initializeDesiredValues(self):
        """Initializes desired values for dynamics and control."""
        self.pdesdhG.value = np.zeros((6, 1))
        self.pdesddbase.value = np.zeros((3, 1))
        self.pdesddq.value = np.zeros((8, 1))
        self.pdesddPSwF.value = np.zeros((6, 1))
        
    def _extractAndSetParameters(self, SwFoot):
        """Extracts parameters from the model and sets Jacobians based on the swing foot."""
        # Extract various parameters and Jacobians from the dynamics computation framework (DCF)
        self.pH.value = self.Dcf.get_H_matrix()
        self.pC_terms.value = self.Dcf.get_C_terms()
        self.ptau.value = self.Dcf.get_tau()
        self.pJacCL.value = self.Dcf.get_CLJacg()
        self.pdJacCLdq.value = self.Dcf.get_CLdJacgdq()
        self.pAG.value = self.Dcf.get_CMM()
        self.pdAGdq.value = self.Dcf.get_dCMMdq()

        jacBase, djacdqBase = self.Dcf.get_jac_base_and_djacdq()
        self.pJacBase.value = jacBase
        self.pdJacBasedq.value = djacdqBase
        
    def _setFootOrientationsAndVelocities(self, SwFoot):
        """Sets foot orientations and velocities based on the swing foot side."""
        # Configure Jacobians and velocities based on which foot is the swing foot
        if SwFoot == -1:  # Left swing foot
            # Stance foot is the right foot
            stance_foot_site = 'right-foot'
            swing_foot_site = 'left-foot'
        elif SwFoot == 1:  # Right swing foot
            # Stance foot is the left foot
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

        # Get pose and velocity for swing and stance foot
        T_SwF, _ = self.Dcf.get_pose_and_vel_for_site(swing_foot_site)
        _, V_SwF = self.Dcf.get_pos_and_vel_for_site(swing_foot_site)
        T_StF, _ = self.Dcf.get_pose_and_vel_for_site(stance_foot_site)
        _, V_StF = self.Dcf.get_pos_and_vel_for_site(stance_foot_site)

        return T_SwF, V_SwF, T_StF, V_StF
    
    def _calculateOrientationErrors(self, T_SwF, V_SwF, T_StF, V_StF, thSwFk_1, des_thSwFk, q, dq, des_x_LIP, s, T):
        """
        Calculates orientation errors for the base and swing foot.

        Parameters:
        - T_SwF: Transformation matrix of the swing foot.
        - V_SwF: Velocity of the swing foot.
        - T_StF: Transformation matrix of the stance foot.
        - V_StF: Velocity of the stance foot.
        - thSwFk_1: Previous orientation angle of the swing foot.
        - des_thSwFk: Desired orientation angle of the swing foot.
        - q: Current configuration of the robot.
        - dq: Current velocity configuration of the robot.
        - des_x_LIP: Desired Linear Inverted Pendulum (LIP) position.

        Returns:
        - Oebase: Orientation error for the base.
        - dOebase: Derivative of the base orientation error.
        - OeSwF: Orientation error for the swing foot.
        - dOeSwF: Derivative of the swing foot orientation error.
        """
        # Calculate the desired base orientation from stance and swing foot positions
        xaxis_SwF = T_SwF[0:2, 0]
        xaxis_StF = T_StF[0:2, 0]
        xaxis_desbase = (xaxis_SwF + xaxis_StF) / 2
        theta_desbase = np.arctan2(xaxis_desbase[1], xaxis_desbase[0])
        quat_desbase = quaternion.Quaternion(axis=[0, 0, 1], angle=theta_desbase)
        
        # Calculate the current base orientation and orientation error
        R_base = quaternion.Quaternion(np.array(q[3:7])).rotation_matrix
        Oebase = R_base.T @ self.QuatOriError(quat_desbase.q, np.array(q[3:7]))

        # Calculate the desired angular velocity for the base
        angvel_desbase = np.array([0, 0, (V_StF[2, 0] + V_SwF[2, 0]) / 2])
        dOebase = dq[3:6] - 0 * angvel_desbase  # Assuming no desired change in angular velocity

        # Calculate the orientation error for the swing foot
        quat_SwFk_1 = quaternion.Quaternion(axis=[0, 0, 1], angle=thSwFk_1)
        quat_SwF_Final = quaternion.Quaternion(axis=[0, 0, 1], angle=des_thSwFk)
        quat_desSwF = quaternion.Quaternion.slerp(quat_SwFk_1, quat_SwF_Final, s)
        quat_SwF = quaternion.Quaternion(matrix=T_SwF[0:3, 0:3])
        OeSwF = self.QuatOriError(quat_desSwF.q, quat_SwF.q)

        # Calculate the derivative of the swing foot orientation error
        dOeSwF = V_SwF[0:3] - np.array([[0, 0, (Angle(des_thSwFk) - Angle(thSwFk_1)).toRadian]]).T / T

        return Oebase, dOebase, OeSwF, dOeSwF
    
    def _computeDynamicsErrors(self, s, PSwFk_1, PSwFk, T, des_comz, des_x_LIP, Oebase, dOebase, OeSwF, dOeSwF, T_SwF, V_SwF):
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
        desired_shape_vector, desired_velocity_vector, hdd_desired =  self._getdeshdh(s, PSwFk_1, PSwFk, T, des_comz, des_x_LIP)
        
        Pcom, Vcom = self.Dcf.get_pcom_vcom()
        
        Pcomxy = des_x_LIP[[0,1]].reshape((2,))
        Vcomxy = des_x_LIP[[2,3]].reshape((2,))
        
        PSwF = T_SwF[0:3,3]
        VSwF = V_SwF[3:6]
        
        # Constructing the current state vectors for position and velocity
        current_shape_vector = np.block([
            [Oebase.reshape((3, 1))],  # Current base orientation error
            [-np.array([[Pcomxy[0], Pcomxy[1], des_comz]]).T + Pcom],  # Current COM position error
            [OeSwF.reshape((3, 1))],  # Current swing foot orientation error
            [PSwF.reshape((3, 1))]  # Current swing foot position error
        ])

        current_velocity_vector = np.block([
            [dOebase.reshape((3, 1))],  # Current base orientation velocity
            [-np.array([[Vcomxy[0], Vcomxy[1], 0]]).T + Vcom],  # Current COM velocity
            [dOeSwF.reshape((3, 1))],  # Current swing foot orientation velocity
            [VSwF.reshape((3, 1))]  # Current swing foot velocity
        ])

        # Calculate shape (position) and velocity errors
        e = desired_shape_vector - current_shape_vector.squeeze()
        de = desired_velocity_vector - current_velocity_vector.squeeze()

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
        kd = 10*np.diag([1,1,0.9,2,2,2,10,10,10,5,5,3])
        kp = (kd/2)**2
        
        # Compute control inputs based on PD control law
        # The control inputs are the desired accelerations to correct position and velocity errors
        self.pdesddh.value = hdd_desired.reshape((12,1)) + kp @ e.reshape((12,1)) + kd @ de.reshape((12,1))
    
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
    
    def _getdeshdh(self, s, PSwFk_1, PSwFk, T, H, x_LIP):
        # Cosine interpolation for the swing foot position
        cos_term = np.cos(np.pi * s)
        swing_foot_interpolation_x = 0.5 * ((1 + cos_term) * PSwFk_1[0] + (1 - cos_term) * PSwFk[0])[0]
        swing_foot_interpolation_y = 0.5 * ((1 + cos_term) * PSwFk_1[1] + (1 - cos_term) * PSwFk[1])[0]
        
        # Parabolic trajectory for z-component centered at s=0.5
        z_trajectory = self.zCL - 4 * self.zCL * (s - 0.5) ** 2
        
        # Constructing desired shape vector
        h = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, swing_foot_interpolation_x, swing_foot_interpolation_y, z_trajectory])
        
        # Derivative of the swing foot position interpolation with respect to time
        dh_x = (0.5 * np.pi * np.sin(np.pi * s) / T) * (PSwFk[0] - PSwFk_1[0])[0]
        dh_y = (0.5 * np.pi * np.sin(np.pi * s) / T) * (PSwFk[1] - PSwFk_1[1])[0]
        dh_z = -8 * self.zCL * (s - 0.5) / T  # Derivative of the z_trajectory with respect to time
        
        # Constructing the derivative of desired shape vector
        dh = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, dh_x, dh_y, dh_z])
        
        # Second derivative calculations based on the linear inverted pendulum model for flat foot walking
        ddh_FF_x = (self.g / H) * (x_LIP[0, 0] - PSwFk_1[0, 0])
        ddh_FF_y = (self.g / H) * (x_LIP[1, 0] - PSwFk_1[1, 0])
        ddh_FF_z = -8 * self.zCL / (T ** 2)  # Second derivative of the z_trajectory with respect to time
        
        # Second derivative effect due to the swing foot position change
        ddh_FF_swing_x = (0.5 * np.pi ** 2 * np.cos(np.pi * s) / (T ** 2)) * (PSwFk[0] - PSwFk_1[0])[0]
        ddh_FF_swing_y = (0.5 * np.pi ** 2 * np.cos(np.pi * s) / (T ** 2)) * (PSwFk[1] - PSwFk_1[1])[0]
        
        # Constructing the second derivative of desired shape vector
        ddh_FF = np.array([0, 0, 0, ddh_FF_x, ddh_FF_y, 0, 0, 0, 0, ddh_FF_swing_x, ddh_FF_swing_y, ddh_FF_z])
        
        return h, dh, ddh_FF
    
    def _extractLIPStatefromFull(self):
        """Extract LIP state [px,py,vx,vy]_com,H from full Digit state.
        par1 q: 37x1 configuration 
        par2 dq: 36x1 vel"""
        p_com, v_com = self.Dcf.get_pcom_vcom()
        return np.array([p_com[0],p_com[1],v_com[0],v_com[1]]),p_com[2]
    
    def SwFspring(self,q,isRightStance):
        """ Extracts the spring compression of the swing leg
        par1 q: 37x1 configuration
        par2 isRightStance: bool True: left swing False: right swing
        """
        if isRightStance: #left swing
            shinSpring = -q[11]
            heelSpring = -q[13]
        else: #right swing
            shinSpring = q[26]
            heelSpring = q[28]

        return np.min([shinSpring,heelSpring])
    
    def handDesqpos(self,swf=0):
        out= np.array([-0.106437-0.3,0.89488,-0.00867,0.44684-0.5, 0.106437+0.3,-0.89488,0.00867,-0.44684+0.5, ])
        return out
    
    "Utility Functions"
    def QuatOriError(self, desQuat, curQuat):
        q_d = quaternion.Quaternion(desQuat)
        R_d = q_d.rotation_matrix
        q_c = quaternion.Quaternion(curQuat)
        R_c = q_c.rotation_matrix
        R_e = 0.5*(R_c.T@R_d - R_d.T@R_c)
        return -R_c@np.array([R_e[2,1],R_e[0,2],R_e[1,0]])
    
    def skew(self, vector):
        return np.array([[0, -vector[2], vector[1]], 
                        [vector[2], 0, -vector[0]], 
                        [-vector[1], vector[0], 0]])
