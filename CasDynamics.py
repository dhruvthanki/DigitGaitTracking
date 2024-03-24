import numpy as np

import os, sys
sys.path.append(os.path.abspath('/home/dhruv/Documents/Github/CommonPlace/DigitModel/CasadiFunctions'))
from C_DigitCasadiFuncs import DigitCasFunc

class DigitCasadiWrapper:
    def __init__(self):
        self.Dcf = DigitCasFunc()
        self.q = np.zeros((37))
        self.dq = np.zeros((36))
        
        self.q_actuated = [7, 8, 9, 10,
                           14, 15,
                           18, 19, 20, 21,
                           22, 23, 24, 25,
                           29, 30,
                           33, 34, 35, 36]
        self.dq_actuated = [x - 1 for x in self.q_actuated]
        
    def get_actuated_indices(self):
        return self.q_actuated, self.dq_actuated
    
    def get_actuated_q(self):
        return self.q[self.q_actuated]

    def set_state(self, q, dq):
        self.q = q
        self.dq = dq

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
        return self.Dcf.Dynamics.CM(self.q, self.dq).full()[0:3]
