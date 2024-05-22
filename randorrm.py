import numpy as np

# Example arrays
motor_torques = np.array([10, 20, 30, 40, 50])  # Replace with your actual motor torques
torque_limits = np.array([100, 200, 300, 400, 500])  # Replace with your actual torque limits

# Calculate the percentage of maximum torque for each motor
percent_to_max_torque = (motor_torques / torque_limits) * 100

print(percent_to_max_torque)
