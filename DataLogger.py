import pickle
import subprocess

class RobotDataLogger:
    def __init__(self, filename='robot_data.pkl'):
        self.filename = filename
        self.data = {
            'time' : [],
            'qpos' : [],
            'qvel' : [],
            'q_desired' : [],
            'dq_desired' : [],
            'ctrl' : [],
            'torso_orientation' : [],
            'swing_foot_orientation' : [],
            'centroidal_angular_momentum' : [],
            'comPos' : [],
            'switching_times' : []
        }
        
    def log_data(self, 
                 time, 
                 qpos, 
                 qvel,
                 q_desired,
                 dq_desired, 
                 ctrl,
                 torso_orientation,
                 swing_foot_orientation,
                 centroidal_angular_momentum,
                 comPos):    
        self.data['time'].append(time)
        self.data['qpos'].append(qpos.copy())
        self.data['qvel'].append(qvel.copy())
        self.data['q_desired'].append(q_desired.copy())
        self.data['dq_desired'].append(dq_desired.copy())
        self.data['ctrl'].append(ctrl.copy())
        self.data['torso_orientation'].append(torso_orientation)
        self.data['swing_foot_orientation'].append(swing_foot_orientation)
        self.data['centroidal_angular_momentum'].append(centroidal_angular_momentum.copy())
        self.data['comPos'].append(comPos.copy())
        
    def saveSwitchingTimes(self, switching_times):
        self.data['switching_times'].append(switching_times)
        
    def save_data(self):
        # Save the list to a pickle file
        with open(self.filename, 'wb') as file:
            pickle.dump(self.data, file)
            print(f"Data saved to {self.filename}")

    def load_data(self):
        # Load the list from a pickle file
        file = open(self.filename, 'rb')
        loaded_data = pickle.load(file)
        return loaded_data

def saveFigure(data, ytitle, filename, layout=None):
    fig = go.Figure(data=data, layout=layout)
    
    # Update layout for better visualization
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title='Time (s)',
        yaxis_title=ytitle,
        legend_title='Legends:',
        legend=dict(
            font=dict(
                size=10  # Change the size to whatever you need
            )
        )
        )
    
    fig.show()
    
    # Save the figure as a PDF file
    pdf_filename = "figures/"+filename+".pdf"
    eps_filename = "figures/"+filename+".eps"
    fig.write_image(pdf_filename, format='pdf')
    time.sleep(2)
    fig.write_image(pdf_filename, format="pdf")
    
    # Convert the PDF to EPS using ghostscript
    subprocess.run(["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=eps2write", f"-sOutputFile={eps_filename}", pdf_filename])
    
def addVerticalLine(data, desired_value):
    # Add the vertical line as a scatter plot for the legend
    data.append(go.Scatter(
        x=[desired_value, desired_value],
        y=[-1, 1],  # None for auto range
        mode='lines',
        name='Impact Detected',
        line=dict(color='Black', width=2, dash='dashdot')
    ))

    
if __name__ == "__main__":
    import time
    import plotly.graph_objects as go
    
    # Assuming RobotDataLogger is a class you've defined to handle data logging
    logger = RobotDataLogger()
    loaded_data = logger.load_data()
    
    # Standing conteller
    jointDictStanding = [
        {"name": 'left-hip-roll',"idx": 0},
        {"name": 'left-hip-yaw',"idx": 1},
        {"name": 'left-hip-pitch',"idx": 2},
        {"name": 'left-knee',"idx": 3},
        {"name": 'left-toe-A',"idx": 4},
        {"name": 'left-toe-B',"idx": 5},
        {"name": 'left-shoulder-roll',"idx": 6},
        {"name": 'left-shoulder-pitch',"idx": 7},
        {"name": 'left-shoulder-yaw',"idx": 8},
        {"name": 'left-elbow',"idx": 9},
        {"name": 'right-hip-roll',"idx": 10},
        {"name": 'right-hip-yaw',"idx": 11},
        {"name": 'right-hip-pitch',"idx": 12},
        {"name": 'right-knee',"idx": 13},
        {"name": 'right-toe-A',"idx": 14},
        {"name": 'right-toe-B',"idx": 15},
        {"name": 'right-shoulder-roll',"idx": 16},
        {"name": 'right-shoulder-pitch',"idx": 17},
        {"name": 'right-shoulder-yaw',"idx": 18},
        {"name": 'right-elbow',"idx": 19}
    ]
    
    jointDict = [
        {"name": 'left-hip-roll',"idx": 0},
        {"name": 'left-hip-yaw',"idx": 1},
        {"name": 'left-hip-pitch',"idx": 2},
        {"name": 'left-knee',"idx": 3},
        {"name": 'left-shoulder-roll',"idx": 4},
        {"name": 'left-shoulder-pitch',"idx": 5},
        {"name": 'left-shoulder-yaw',"idx": 6},
        {"name": 'left-elbow',"idx": 7},
        {"name": 'right-hip-roll',"idx": 8},
        {"name": 'right-hip-yaw',"idx": 9},
        {"name": 'right-hip-pitch',"idx": 10},
        {"name": 'right-knee',"idx": 11},
        {"name": 'right-shoulder-roll',"idx": 12},
        {"name": 'right-shoulder-pitch',"idx": 13},
        {"name": 'right-shoulder-yaw',"idx": 14},
        {"name": 'right-elbow',"idx": 15}
    ]
    
    # suffix = "ExternalDisturbance"
    suffix = "GaitTracking"
    # suffix = ""
    
    idx = 3
    data=[]
    data.append(go.Scatter(x=[entry[idx] for entry in loaded_data['qpos']], 
                           y=[entry[idx] for entry in loaded_data['qvel']], 
                           mode='lines', 
                           name="Left Knee Joint Limit Cycle"))
    ytitle = 'Knee Joint Limit Cycle'
    saveFigure(data, ytitle, "KneeJointLimitCycle"+suffix)
    
    # data=[]
    # for joint in jointDict:
    #     data.append(go.Scatter(x=loaded_data['time'], y=[entry[joint["idx"]] for entry in loaded_data['qpos']], mode='lines', name=joint["name"]))
    # # data.append(go.Scatter(
    # #     x=[min(loaded_data['time']), max(loaded_data['time'])],
    # #     y=[0.5, 0.5],
    # #     mode='lines',
    # #     name='Desired Value',
    # #     line=dict(color='Red', width=2, dash='dashdot')
    # # ))
    # ytitle = 'Joint-space Trajectory Tracking Error (deg)'
    # saveFigure(data, ytitle, "JointSpaceTrajectoryTrackingError"+suffix)
    
    # data=[]
    # for joint in jointDictStanding:
    #     data.append(go.Scatter(x=loaded_data['time'], y=[entry[joint["idx"]] for entry in loaded_data['ctrl']], mode='lines', name=joint["name"]))
    # for line in loaded_data['switching_times']:
    #     addVerticalLine(data, line)
    # ytitle = 'Percentage of Maximum Torque limit (%)'
    # saveFigure(data, ytitle, "TorqueLimitPercentage"+suffix)
    
    # data=[
    #     go.Scatter(x=loaded_data['time'], y=[entry[0] for entry in loaded_data['swing_foot_orientation']], mode='lines', name='Roll'), 
    #     go.Scatter(x=loaded_data['time'], y=[entry[1] for entry in loaded_data['swing_foot_orientation']], mode='lines', name='Pitch'), 
    #     go.Scatter(x=loaded_data['time'], y=[entry[2] for entry in loaded_data['swing_foot_orientation']], mode='lines', name='Yaw')
    #     ]
    # ytitle = 'Swing Foot Orientation (deg)'
    # saveFigure(data, ytitle, "SwingFootOrientation"+suffix)
    
    # rolls = [entry[0] for entry in loaded_data['torso_orientation']]
    # # # rolls = [roll - 180 for roll in rolls if roll > 0]
    # # # rolls = [roll + 180 for roll in rolls if roll < 0]
    # rollis = [roll - 180 if roll > 0 else roll + 180 if roll < 0 else roll for roll in rolls]
    # data=[
    #     go.Scatter(x=loaded_data['time'], y=rollis, mode='lines', name='Roll'), 
    #     go.Scatter(x=loaded_data['time'], y=[entry[1] for entry in loaded_data['torso_orientation']], mode='lines', name='Pitch'), 
    #     go.Scatter(x=loaded_data['time'], y=[entry[2] for entry in loaded_data['torso_orientation']], mode='lines', name='Yaw')
    #     ]
    # ytitle = 'Torso Orientation (deg)'
    # saveFigure(data, ytitle, "TorsoOrientation"+suffix)
    
    # data=[
    #     go.Scatter(x=loaded_data['time'], y=[entry[0] for entry in loaded_data['comPos']], mode='lines', name='X'),
    #     go.Scatter(x=loaded_data['time'], y=[entry[1] for entry in loaded_data['comPos']], mode='lines', name='Y'),
    #     go.Scatter(x=loaded_data['time'], y=[entry[2] for entry in loaded_data['comPos']], mode='lines', name='Z')
    #     ]
    # ytitle = 'Center of Mass position (m)'
    # saveFigure(data, ytitle, "CenterOfMassPosition"+suffix)
    
    # data=[
    #     go.Scatter(x=loaded_data['time'], y=[entry[0] for entry in loaded_data['centroidal_angular_momentum']], mode='lines', name='X'),
    #     go.Scatter(x=loaded_data['time'], y=[entry[1] for entry in loaded_data['centroidal_angular_momentum']], mode='lines', name='Y'),
    #     go.Scatter(x=loaded_data['time'], y=[entry[2] for entry in loaded_data['centroidal_angular_momentum']], mode='lines', name='Z')
    #     ]
    # ytitle = 'Centroidal Angular Momentum (kg.m^2/s)'
    # saveFigure(data, ytitle, "CentroidalAngularMomentum"+suffix)
