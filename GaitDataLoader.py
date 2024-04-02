import numpy as np
import scipy.io as spio
from scipy.special import comb
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class GaitDataLoader:
    def __init__(self, data_path):
        self.gait_data = spio.loadmat(data_path)['data']
        self.t_step = self.gait_data['final_time'][0][0][0][0]
        self.alpha = self.process_alpha_indices(self.gait_data['a'][0, 0])
        self.combinations, self.d_combinations, self.dd_combinations = self.precompute_combination_terms(self.alpha.shape[1] - 1)
        
        self.p_com = self.gait_data['p_com'][0, 0]
        self.com_x_bezier_alpha = self.fit_bezier_curve(self.p_com[0, :], self.alpha.shape[1] - 1)
        self.com_y_bezier_alpha = self.fit_bezier_curve(self.p_com[1, :], self.alpha.shape[1] - 1)
        self.com_z_bezier_alpha = self.fit_bezier_curve(self.p_com[2, :], self.alpha.shape[1] - 1)

    @staticmethod
    def process_alpha_indices(alpha):
        indices = np.array([0, 1, -2, 3, # left leg
                            4, 5, # right toe A and B
                            6, -7, 8, 9, # left hand
                            10, 11, -12, 13, # right leg
                            14, 15, # right toe A and B
                            16, -17, 18, 19]) # right hand
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
    
    def bezier_curve(self, t, control_points):
        """Calculate the Bézier curve point for a given parameter t and control points."""
        n = len(control_points) - 1
        point = np.zeros_like(control_points[0])
        for i, p in enumerate(control_points):
            bernstein_poly = (np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))) * (t ** i) * ((1 - t) ** (n - i))
            point += bernstein_poly * p
        return point
    
    def bezier_curve_derivative(self, t, control_points):
        """Calculate the derivative of the Bézier curve at a given parameter t."""
        n = len(control_points) - 1
        derivative = np.zeros_like(control_points[0])
        for i, p in enumerate(control_points[:-1]):
            bernstein_poly = n * ((np.math.factorial(n - 1) / (np.math.factorial(i) * np.math.factorial(n - i - 1))) * (t ** i) * ((1 - t) ** (n - i - 1)))
            derivative += bernstein_poly * (control_points[i + 1] - control_points[i])
        return derivative

    def fit_bezier_curve(self, data, degree):
        """Fit a Bézier curve of a given degree to 1D data and its derivative."""
        # Initial guess for the control points
        initial_guess = np.linspace(min(data), max(data), degree + 1)
        
        # Time parameters for data points, assuming equally spaced
        t_values = np.linspace(0, 1, len(data))
        
        def objective_function(control_points):
            # Calculate the difference between the Bézier curve and the data points
            bezier_points = np.array([self.bezier_curve(t, control_points) for t in t_values])
            return bezier_points - data
        
        # Solve the least squares problem
        result = least_squares(objective_function, initial_guess)
        
        return result.x
    
    def getCOMTrajectory(self, phase):
        comX = self.bezier_curve(phase, self.com_x_bezier_alpha)
        comY = self.bezier_curve(phase, self.com_y_bezier_alpha)
        comZ = self.bezier_curve(phase, self.com_z_bezier_alpha)
        
        velX = self.bezier_curve_derivative(phase, self.com_x_bezier_alpha)
        velY = self.bezier_curve_derivative(phase, self.com_y_bezier_alpha)
        velZ = self.bezier_curve_derivative(phase, self.com_z_bezier_alpha)
        return np.array([comX, comY, comZ]), np.array([velX, velY, velZ])

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

def plot_bezier_curve_with_derivatives(control_points, data, num_points=100):
    """
    Plot a Bezier curve and its first and second derivatives.
    """
    t_values = np.linspace(0, 1, num_points)
    curve_values = [GaitDataLoader.bezier_curve(0, t, control_points) for t in t_values]

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, curve_values, label='Bezier Curve')
    plt.scatter(t_values, data, color='red', label='Data Points')
    plt.title('Bezier Curve and its Derivatives')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
if __name__ == '__main__':
    gait_data_loader = GaitDataLoader('Digit-data_12699.mat')
    # plot_gait_data(gait_data_loader)
        
    plot_bezier_curve_with_derivatives(gait_data_loader.alpha[3,:], 
                                       gait_data_loader.gait_data['pos'][0,0][9, :], 
                                       gait_data_loader.gait_data['pos'][0,0][9, :].shape[0])
    
    plot_bezier_curve_with_derivatives(gait_data_loader.alpha[13,:], 
                                       gait_data_loader.gait_data['pos'][0,0][24, :], 
                                       gait_data_loader.gait_data['pos'][0,0][24, :].shape[0])
    
    # plot_bezier_curve_with_derivatives(gait_data_loader.com_x_bezier_alpha, gait_data_loader.p_com[0, :], gait_data_loader.p_com.shape[1])
    # plot_bezier_curve_with_derivatives(gait_data_loader.com_y_bezier_alpha, gait_data_loader.p_com[1, :], gait_data_loader.p_com.shape[1])
    # plot_bezier_curve_with_derivatives(gait_data_loader.com_z_bezier_alpha, gait_data_loader.p_com[2, :], gait_data_loader.p_com.shape[1])
    
    # t_values = np.linspace(0, 1, gait_data_loader.p_com.shape[1])
    # curve_valuesX = [GaitDataLoader.bezier_curve(0, t, gait_data_loader.com_x_bezier_alpha) for t in t_values]
    # curve_valuesY = [GaitDataLoader.bezier_curve(0, t, gait_data_loader.com_y_bezier_alpha) for t in t_values]
    # curve_valuesZ = [GaitDataLoader.bezier_curve(0, t, gait_data_loader.com_z_bezier_alpha) for t in t_values]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatter plot
    # ax.plot(gait_data_loader.p_com[0, :], gait_data_loader.p_com[1, :], gait_data_loader.p_com[2, :])
    # ax.plot(curve_valuesX, curve_valuesY, curve_valuesZ, label='Bezier Curve')

    # # Setting labels
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    
    # # Getting the limits for each axis and calculating the center
    # x_limits = ax.get_xlim()
    # y_limits = ax.get_ylim()
    # z_limits = ax.get_zlim()

    # x_center = np.mean(x_limits)
    # y_center = np.mean(y_limits)
    # z_center = np.mean(z_limits)

    # # Calculating the maximum range between the limits
    # max_range = max(np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)) / 2

    # # Setting the limits for each axis to the same range
    # ax.set_xlim(x_center - max_range, x_center + max_range)
    # ax.set_ylim(y_center - max_range, y_center + max_range)
    # ax.set_zlim(z_center - max_range, z_center + max_range)

    # Displaying the plot
    plt.show()
