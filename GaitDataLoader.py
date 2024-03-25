import numpy as np
import scipy.io as spio
from scipy.special import comb
import matplotlib.pyplot as plt

class GaitDataLoader:
    def __init__(self, data_path):
        self.gait_data = spio.loadmat(data_path)['data']
        self.t_step = self.gait_data['final_time']
        self.alpha = self.process_alpha_indices(self.gait_data['a'][0, 0])
        self.combinations, self.d_combinations, self.dd_combinations = self.precompute_combination_terms(self.alpha.shape[1] - 1)

    @staticmethod
    def process_alpha_indices(alpha):
        indices = np.array([0, 1, -2, 3, 4, 5, 10, 11, -12, 13, 14, 15, 6, -7, 8, 9, 16, -17, 18, 19])
        return np.array([alpha[abs(idx)] if idx >= 0 else -alpha[abs(idx)] for idx in indices]).reshape(20, -1)

    @staticmethod
    def precompute_combination_terms(degree):
        combinations = comb(degree, np.arange(degree + 1), exact=False)
        d_combinations = degree * comb(degree - 1, np.arange(degree), exact=False)
        dd_combinations = degree * (degree - 1) * comb(degree - 2, np.arange(degree - 1), exact=False)
        return combinations, d_combinations, dd_combinations

    def evaluate_bezier_curve(self, phase):
        degree = self.alpha.shape[1] - 1
        s_powers = np.power(phase, np.arange(degree + 1))
        one_minus_s_powers = np.power(1 - phase, np.arange(degree, -1, -1))
        
        bezier_basis = self.combinations * s_powers * one_minus_s_powers
        q_des = np.dot(self.alpha, bezier_basis)

        d_bezier_basis = self.d_combinations * np.power(phase, np.arange(degree)) * np.power(1 - phase, np.arange(degree - 1, -1, -1))
        dq_des = np.dot(self.alpha[:, 1:] - self.alpha[:, :-1], d_bezier_basis) / self.t_step

        dd_bezier_basis = self.dd_combinations * np.power(phase, np.arange(degree - 1)) * np.power(1 - phase, np.arange(degree - 2, -1, -1))
        ddq_des = np.dot((self.alpha[:, 2:] - 2 * self.alpha[:, 1:-1] + self.alpha[:, :-2]), dd_bezier_basis) / (self.t_step ** 2)
        
        return q_des.squeeze(), dq_des.squeeze(), ddq_des.squeeze()
    
    def evaluate_bezier_curve_relabelled(self, phase):
        q_des, dq_des, ddq_des = self.evaluate_bezier_curve(phase)
        return q_des.squeeze(), dq_des.squeeze(), ddq_des.squeeze()

def plot_gait_data(gait_data_loader: GaitDataLoader):
    rows = gait_data_loader.alpha.shape[0]
    phase = gait_data_loader.gait_data['tau_time'][0, 0]
    ncp = phase.shape[1]
    
    time = np.linspace(0, 1, ncp)
    q_des, dq_des, ddq_des = np.zeros((rows, ncp)), np.zeros((rows, ncp)), np.zeros((rows, ncp))
    for t in range(ncp):
        q_des[:, t], dq_des[:, t], ddq_des[:, t] = gait_data_loader.evaluate_bezier_curve(time[t])

    for descriptor, data in zip(['pos', 'vel', 'acc'], [q_des, dq_des, ddq_des]):
        fig, ax = plt.subplots()
        ax.plot(phase[0, :], data[3, :], label='Bezier')
        ax.plot(phase[0, :], gait_data_loader.gait_data[descriptor][0,0][9, :], label='Digit')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    gait_data_loader = GaitDataLoader('Digit-data_12699.mat')
    plot_gait_data(gait_data_loader)
