import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open('my_list.pkl', 'rb') as file:
    loaded_list = pickle.load(file)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.scatter([obj[0] for obj in loaded_list['mjdata']], [obj[1] for obj in loaded_list['mjdata']], [obj[2] for obj in loaded_list['mjdata']], label='com')
    plt.scatter([obj[0] for obj in loaded_list['desired']], [obj[1] for obj in loaded_list['desired']], [obj[2] for obj in loaded_list['desired']], label='comdes')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.legend()
    plt.show()