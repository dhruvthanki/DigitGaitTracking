"""
Written by: Dhruv Thanki (thankid@udel.edu)
"""
import casadi as cs
import numpy as np
import time

class LIPMPC():
    def __init__(self,PData):
        self.H = PData['H']
        self.N = 5
        self.g = 9.80665
        self.beta = np.sqrt(self.g/self.H)
        self.Tst = 0.3
        self.DObs = False

        self.opti = cs.Opti()
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)
        self.opti.solver("ipopt", p_opts, s_opts)
        # self.opti.solver("sqpmethod", p_opts)
        
        # Dynamic Obstacle MPC parameter
        if(self.DObs==True):
            self.pos_DO = self.opti.parameter(2,self.N+1) #footstep position of LIP (px,py)_footstep
            self.DOpos0 = np.array([2, 0])
            self.DOvel = np.array([0.5, 0])
            self.gamma = 0.5
            self.rDO = 0.5

        self.A = self._A(self.Tst)
        self.B = self._B(self.Tst)

        # Reachability Bounds for Left stance and Right stance (ux, uy)
        self.ls_ub_u = np.array([[0.6],[-0.1]])
        self.ls_lb_u = np.array([[-0.2],[-0.5]])
        self.rs_ub_u = np.array([[0.6],[0.5]])
        self.rs_lb_u = np.array([[-0.2],[0.1]])

        #MPC QP paramters
        self.ub_u = self.opti.parameter(2,self.N)
        self.lb_u = self.opti.parameter(2,self.N)
        self.x0 = self.opti.parameter(4)
        self.h0 = self.opti.parameter(1)

        # MPC QP variables
        self.x = self.opti.variable(4,self.N+1)
        self.h = self.opti.variable(1,self.N+1)
        self.u = self.opti.variable(2,self.N)
        self.v = self.opti.variable(1,self.N)

        self.xguess_nextrun = np.zeros((4,self.N+1))
        self.hguess_nextrun = np.zeros((1,self.N+1))
        self.uguess_nextrun = np.zeros((2,self.N))
        self.vguess_nextrun = np.zeros((1,self.N))

        # Constants
        self.v_max = 0.3
        self.dtheta_max = np.deg2rad(10)
        self.dcom = 0.3

        # Desired Trajectory Parametrized by theta
        self.theta_0 = self.opti.variable(1)
        self.theta_0guess_nextrun = 0
        self.des_ref = []
        self.des_ori = []
        for k in range(0, self.N+1, 1):
            if k==0:
                theta_k = self.theta_0
            elif k>0:
                theta_k += self.v[k-1]
            des_ref_pos, _, des_ref_heading = self.calc_des(theta_k)
            self.des_ref = cs.horzcat(self.des_ref, des_ref_pos)
            self.des_ori = cs.horzcat(self.des_ori, des_ref_heading)

        # Formulate MPC QP
        self._formulateMPCC()
        self.fc_theta_actual = 0
        self._find_closest()

    def _find_closest(self):
        self.fc_opti = cs.Opti()
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)
        self.fc_opti.solver("ipopt", p_opts, s_opts)
        self.fc_v_theta0 = self.fc_opti.variable(1)
        self.fc_p_theta0 = self.fc_opti.parameter(1)
        des_pos,_,_ = self.calc_des(self.fc_v_theta0)
        self.fc_p_xpos = self.fc_opti.parameter(2)
        err = self.fc_p_xpos - des_pos
        self.fc_opti.minimize( cs.dot(err, err) )
        bb = 0.5
        self.fc_opti.subject_to(self.fc_opti.bounded(self.fc_p_theta0-bb, self.fc_v_theta0, self.fc_p_theta0+bb))
        self.fc_opti.subject_to(self.fc_v_theta0>=0)

    def Solve_fc(self, itheta0=None, ixpos=None):
        self.fc_opti.set_value(self.fc_p_theta0, itheta0)
        self.fc_opti.set_value(self.fc_p_xpos, ixpos[:2])

        self.fc_solution = self.fc_opti.solve()
        
        theta = self.fc_solution.value(self.fc_v_theta0)
        return theta

    # # Circular Trajectory
    def calc_des(self, time):
        omega = 1
        A = 2
        ref_pos = cs.vertcat(A*cs.sin(omega*time), -A*cs.cos(omega*time) + A)
        ref_vel = cs.vertcat(A*omega*cs.cos(omega*time), A*omega*cs.sin(omega*time))
        # ref_ori = cs.atan2(ref_vel[1], ref_vel[0])#
        ref_ori = omega*time#
        return ref_pos, ref_vel, ref_ori

    # # Sine Trajectory
    # def calc_des(self, time):
    #     omega = 1
    #     A = 2
    #     ref_pos = cs.vertcat(A*omega*time, A*cs.sin(omega*time))
    #     ref_vel = cs.vertcat(A*omega, A*omega*cs.cos(omega*time))
    #     ref_ori = cs.atan2(A*omega*cs.cos(omega*time),A*omega)
    #     return ref_pos, ref_vel, ref_ori
    
    # Line Trajectory
    # def calc_des(self, time):
    #     ref_pos = cs.vertcat(time, 0)
    #     ref_vel = 0
    #     ref_ori = 0
    #     return ref_pos, ref_vel, ref_ori

    # Inplace Walking
    # def calc_des(self,time):
    #     ref_pos = cs.vertcat(0, 0)
    #     ref_vel = 0
    #     ref_ori = time
    #     return ref_pos, ref_vel, ref_ori

    def _add_DynObsConstraint(self):
        for k in range(1, self.N+1, 1):
            hk_xk = cs.dot(self.x[:2,k] - self.pos_DO[:,k], self.x[:2,k] - self.pos_DO[:,k]) - (self.rDO + 0.25)**2
            hk_xk1 = cs.dot(self.x[:2,k-1] - self.pos_DO[:,k-1], self.x[:2,k-1] - self.pos_DO[:,k-1]) - (self.rDO + 0.25)**2
            self.opti.subject_to(hk_xk >= (1 - self.gamma)*hk_xk1)

    def predictTouchDownLIPState(self, x_intermediate,s,T,PStFabs):
        """Predicts touch down state of LIP when an intermediate state of LIP is given and its progress s
        par1 x_intermediate: intermediate state of the LIP
        par2 s: progress in the step [0,1]
        par3 T: Timestep of the step
        par4 PStFabs: Absolute postion of the current footstep"""
        T_end = (1-s)*T
        A_abs = self._A_abs(T_end)
        B_abs = self._B(T_end)
        x_end = A_abs@x_intermediate + B_abs@PStFabs[0:2]
        return x_end

    def _CLSolution_abs(self,x0,t,pfst_abs):
        return self._A_abs(t)@x0+self._B(t)@pfst_abs
        
    def _A_abs(self, t): #when foot position is absolute x-p
        shbt = np.sinh(self.beta*t)
        bshbt = self.beta*shbt
        shbt_b = shbt/self.beta
        chbt = np.cosh(self.beta*t)
        return np.array([[chbt,0,shbt_b,0],
                        [0,chbt,0,shbt_b],
                        [bshbt,0,chbt,0],
                        [0,bshbt,0,chbt]])

    def _A(self,t):
        shbt_b = np.sinh(self.beta*t)/self.beta
        chbt = np.cosh(self.beta*t)
        return np.array([[1,0,shbt_b,0],
                [0,1,0,shbt_b],
                [0,0,chbt,0],
                [0,0,0,chbt]])

    def _B(self,t):
        chbt = np.cosh(self.beta*t)
        bshbt = self.beta*np.sinh(self.beta*t)
        return np.array([[1-chbt,0],
                        [0,1-chbt],
                        [-bshbt,0],
                        [0,-bshbt]])
    
    def _addDynamicsConstraint(self):
        for k in range(0, self.N, 1):
            self.opti.subject_to(self.x[:,k+1]==self.A@self.x[:,k]+self.B@self.u[:,k])
            self.opti.subject_to(self.opti.bounded(-self.dtheta_max,self.h[k+1]-self.h[k],self.dtheta_max))
    
    def _addInitialStateConstraint(self):
        self.opti.subject_to(self.x[:,0]==self.x0)
        self.opti.subject_to(self.h[0]==self.h0)
    
    def _addReachabilityConstraints(self):
        for k in range(self.N):
            cth = cs.cos(self.h[k])
            sth = cs.sin(self.h[k])
            des_Rk = cs.vertcat(cs.horzcat(cth, sth), cs.horzcat(-sth, cth))
            self.opti.subject_to(self.opti.bounded(self.lb_u[:,k], des_Rk@self.u[:,k] ,self.ub_u[:,k]))

    def _add_state_bounds(self):
        for k in range(self.N):
            dx = self.x[:,k+1] - self.x[:,k]
            dist = cs.dot(dx[:2], dx[:2])
            self.opti.subject_to(dist <= (self.dcom)**2 )
            
            vel = cs.dot(self.x[2:4,k+1], self.x[2:4,k+1])
            self.opti.subject_to(vel <= 1)

    def _add_v_bounds(self):
        self.opti.subject_to(self.theta_0 >= 0)
        for k in range(self.N):
            self.opti.subject_to(self.opti.bounded(0.0, self.v[k], self.v_max))

    def Solve(self, timek=None, x0=None, heading=None, lf_first=None):
        
        if(self.DObs==True):
            for k in range(0, self.N+1, 1):
                DObsP = self.DOpos0 + self.DOvel*(timek + k*self.Tst)
                self.opti.set_value(self.pos_DO[:,k], DObsP)
        
        ls = lf_first
        for k in range(0, self.N, 1):
            if ls:
                self.opti.set_value(self.ub_u[:,k], self.ls_ub_u)
                self.opti.set_value(self.lb_u[:,k], self.ls_lb_u)
            else:
                self.opti.set_value(self.ub_u[:,k], self.rs_ub_u)
                self.opti.set_value(self.lb_u[:,k], self.rs_lb_u)
            ls = not ls
        
        # Set Initial State Parameter
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.h0, heading)

        # Set Iniial guess of the variables
        self.opti.set_initial(self.theta_0, self.theta_0guess_nextrun)
        self.opti.set_initial(self.x, self.xguess_nextrun)
        self.opti.set_initial(self.h, self.hguess_nextrun)
        self.opti.set_initial(self.u, self.uguess_nextrun)
        self.opti.set_initial(self.v, self.vguess_nextrun)

        # Solve MPC QP
        t0 = time.time()
        self.solution = self.opti.solve()
        t1 = time.time()
        t = t1 - t0
        # print(t)

        # validate theta
        self.fc_theta_actual = self.Solve_fc(self.fc_theta_actual, x0)
        err_theta = self.solution.value(self.theta_0) - self.fc_theta_actual
        # print(err_theta)

        self.theta_0guess_nextrun = self.solution.value(self.theta_0)
        self.xguess_nextrun = self.solution.value(self.x)
        self.hguess_nextrun = self.solution.value(self.h)
        self.uguess_nextrun = self.solution.value(self.u)
        self.vguess_nextrun = self.solution.value(self.v)

        u0 = self.solution.value(self.u)[:,0]
        nheading = self.solution.value(self.h)[1]
        return u0, nheading, self.xguess_nextrun, self.uguess_nextrun, self.hguess_nextrun, self.theta_0guess_nextrun

    def _formulateMPCC(self):
        #constraints
        self._addDynamicsConstraint()
        self._addInitialStateConstraint()
        self._addReachabilityConstraints()
        self._add_state_bounds()
        self._add_v_bounds()
        if(self.DObs==True):
            self._add_DynObsConstraint()

        #cost
        runningcost_pos = 0
        runningcost_v = 0
        runningcost_heading = 0
        runningcost_u = 0

        # Optimize theta_0 such that the desired position has minimum dist from x0
        theta_err = self.des_ref[:,0] - self.x0[:2]
        theta_error = cs.dot(theta_err, theta_err)

        w_u = np.array([[10, 0], [0, 10]])
        runningcost_u += cs.dot(self.u[:,0], w_u@self.u[:,0])

        runningcost_v += cs.dot(self.v[0], self.v[0])

        w = np.array([[0.5, 0], [0, 0.1]])
        # w = np.array([[0.77, 0], [0, 5]])
        for i in range(1, self.N, 1):
            heading_err = self.h[i] - self.des_ori[i]
            runningcost_heading += cs.dot(heading_err, heading_err)

            lead_lag_err = cs.vertcat(  cs.horzcat(cs.sin(self.des_ori[i]), -cs.cos(self.des_ori[i])),
                                        cs.horzcat(-cs.cos(self.des_ori[i]), -cs.sin(self.des_ori[i])))
            err_pos = self.x[0:2,i] - self.des_ref[:,i]
            err = lead_lag_err@err_pos
            runningcost_pos += cs.dot(err,w@err)
            
            runningcost_u += cs.dot(self.u[:,i], w_u@self.u[:,i])

            runningcost_v += cs.dot(self.v[i], self.v[i])

        # Terminal Heading Cost
        heading_err = self.h[self.N] - self.des_ori[self.N]
        runningcost_heading += cs.dot(heading_err, heading_err)

        # Terminal position cost
        lead_lag_err = cs.vertcat(  cs.horzcat(cs.sin(self.des_ori[self.N]), -cs.cos(self.des_ori[self.N])),
                                    cs.horzcat(-cs.cos(self.des_ori[self.N]), -cs.sin(self.des_ori[self.N])))
        err_pos = self.x[0:2,self.N] - self.des_ref[:,self.N]
        err = lead_lag_err@err_pos
        runningcost_pos += cs.dot(err,w@err)
        
        v_cost_w = 30 #30
        # Total Cost
        self.opti.minimize(runningcost_pos - v_cost_w*runningcost_v + runningcost_u + 50*runningcost_heading + 100*theta_error)