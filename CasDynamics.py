import numpy as np

import os, sys
sys.path.append(os.path.abspath('submodules/CommonPlace/DigitModel/CasadiFunctions'))
from C_DigitCasadiFuncs import DigitCasFunc

class DigitCasadiWrapper:
    def __init__(self):
        self.Dcf = DigitCasFunc()
        self.q = np.zeros((37))
        self.dq = np.zeros((36))
        
        self.q_actuated = [7, 8, 9, 10, # left leg
                           14, 15, # left toe A and B
                           18, 19, 20, 21, # left hand
                           22, 23, 24, 25, # right leg
                           29, 30, # right toe A and B
                           33, 34, 35, 36] # right hand
        self.dq_actuated = [x - 1 for x in self.q_actuated]
        
    def get_actuated_indices(self):
        return self.q_actuated, self.dq_actuated
    
    def get_actuated_q(self):
        return self.q[self.q_actuated]

    def set_state(self, q, dq):
        self.q = q
        self.dq = dq
        
    def get_state(self):
        return self.q, self.dq

    def get_B_matrix(self):
        return self.Dcf.Dynamics.B()['o0'].full()
    
    def get_H_matrix(self):
        return self.Dcf.Dynamics.H_matrix(self.q).full()

    def get_C_terms(self):
        return self.Dcf.Dynamics.C_terms(self.q, self.dq).full()

    def get_tau_damp(self):
        return self.Dcf.Dynamics.tau_damp(self.dq).full()

    def get_tau_stiff(self):
        return self.Dcf.Dynamics.tau_stiff(self.q).full()[1::]  # Not including friction
    
    def get_tau(self):
        return self.get_tau_damp() + self.get_tau_stiff()

    def get_CLJacg(self):
        return self.Dcf.HolonomicCons.CLJacg(self.q).full()

    def get_CLdJacgdq(self):
        return self.Dcf.HolonomicCons.CLdJacgdq(self.q, self.dq).full()

    def get_jac_and_djac_for_site(self, site):
        if site == 'left-foot':
            jac = self.Dcf.Site.Jac.leftfoot(self.q).full()
            djacdq = self.Dcf.Site.dJacdq.leftfoot(self.q, self.dq).full()
        elif site == 'right-foot':
            jac = self.Dcf.Site.Jac.rightfoot(self.q).full()
            djacdq = self.Dcf.Site.dJacdq.rightfoot(self.q, self.dq).full()
        else:
            raise ValueError("Invalid site name. Must be 'leftfoot' or 'rightfoot'.")
        return jac, djacdq

    def get_CMM(self):
        return self.Dcf.Dynamics.CMM(self.q, self.dq).full()

    def get_dCMMdq(self):
        return self.Dcf.Dynamics.dCMMdq(self.q, self.dq).full()
    
    def get_jac_base_and_djacdq(self):
        jac_base = self.Dcf.Body.Jac.br(self.q).full()[0:3, :]
        djacdq_base = self.Dcf.Body.dJacdq.br(self.q, self.dq).full()[0:3, :]
        return jac_base, djacdq_base
    
    def get_pcom_vcom(self):
        pcom = self.Dcf.Dynamics.pcom(self.q).full()
        vcom = self.Dcf.Dynamics.vcom(self.q, self.dq).full()
        return pcom, vcom

    def get_pose_and_vel_for_site(self, site):
        if site == 'left-foot':
            pose = self.Dcf.Site.Pose.leftfoot(self.q).full()
            vel = self.Dcf.Site.Vel.leftfoot(self.q, self.dq).full()
        elif site == 'right-foot':
            pose = self.Dcf.Site.Pose.rightfoot(self.q).full()
            vel = self.Dcf.Site.Vel.rightfoot(self.q, self.dq).full()
        else:
            raise ValueError("Invalid site name. Must be 'leftfoot' or 'rightfoot'.")
        return pose, vel
 
    def get_pos_and_vel_for_site(self, site):
        pose, vel = self.get_pose_and_vel_for_site(site)
        return pose[0:3,3], vel
    
    def get_CM(self):
        return self.Dcf.Dynamics.CM(self.q, self.dq).full()
    
    def get_u_limit(self):
        self.u_limit = np.array([[1.4,   1.4,  12.5,  12.5,   0.9,   0.9,   1.4,   1.4,   1.4,
                          1.4,   1.4,   1.4,  12.5,  12.5,   0.9,   0.9,   1.4,   1.4,
                          1.4,   1.4]]).T
        # self.u_limit = np.array([[1.4,   1.4,  12.5,  12.5,   0.9,   0.9,   1.4,   1.4,   1.4,
        #                   1.4,   1.4,   1.4,  12.5,  9,   0.5,   0.5,   1.4,   1.4,
        #                   1.4,   1.4]]).T
        return self.u_limit
        
    def set_u_limit(self, u_limit):
        self.u_limit = u_limit
        pass
