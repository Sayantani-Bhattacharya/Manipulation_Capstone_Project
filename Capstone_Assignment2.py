import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
import math
import time


'''
Instruction for usage: User can simply run the code and the program will generate a .csv file which can be used for simulation.
'''  

class Motion():

    def __init__(self):
        self.frequency = 0.01
        self.robot_state_trajectory = []

    def NextState(self, current_config, control_param, timestep=0.01, max_angular_speed = 10):
        '''
        Input:
        current_config (12 - np array) : representing the current configuration of the robot 
                    1. arm joint angles(5 vals)
                    2. wheel angles(4 vals)
                    3. chasis configuration(3 vals) -> (phi, x, y):                        
                        phi: angle of the base
                        (x, y): location of its center. 

        control_param (9-np array) : representing the controls
                    1. arm joint speeds (5 variables).
                    2. wheel speeds (4 variables).                     

        timestep (int) : The time interval for updation of the Euler function.

        max_angular_speed (float) : The maximum angular speed of the arm joints and the wheels. 
                                    For example, if this value is 12.3, the angular speed of the wheels and arm joints is limited to the range [-12.3 radians/s, 12.3 radians/s]. 
                                    Any speed in the 9-vector of controls that is outside this range will be set to the nearest boundary of the range. If you don't want speed limits, 
                                    use a very large number.
        
        The function is based on a simple first-order Euler step,      
        i.e.,
        new arm joint angles = (old arm joint angles) + (joint speeds) * Δ t
        new wheel angles = (old wheel angles) + (wheel speeds) * Δ t 
        new chassis configuration is obtained from odometry.


        Returns (np array of shape N*12):

                Each row would have followeing data:
                    1. new arm joint angles(5 vals)
                    2. new wheel angles(4 vals)
                    3. new chasis configuration(3 vals) -> (phi, x, y):                        
                        phi: angle of the base
                        (x, y): location of its center.

                Each row representing that instant of robot state and N such rows would be returned.                
                where, N : no of iterations of config values
        '''

        # max_speed step.
        for val in control_param:
            if val < (-1 * max_angular_speed):
                val = (-1 * max_angular_speed)
            elif val > max_angular_speed:
                val = max_angular_speed
            # substitute this val in control_param

        next_state = []
        # Iteration step.
        chasis_config = current_config[0:3]

        joint_angles = current_config[3:8]
        joint_speed = control_param[0:5]
        joint_angle_next = joint_angles + joint_speed * timestep

        wheel_angles = current_config[8:12]
        wheel_speed = control_param[5:9]
        wheel_angle_next = wheel_angles + wheel_speed * timestep  

        u = wheel_angle_next
        # should u be wheel_angle_next - wheel_angle ?
        q = chasis_config
        chasis_config_next = []
        radius = 0.0475                     
        l = 0.235                     
        w = 0.15
        H = 1/radius * np.array([[-l-w, 1, -1],
                                [l+w, 1, 1],
                                [l+w, 1, -1],
                                [-l-w, 1, 1]])
        F = np.linalg.pinv(H)  # see if inv or pinv
        # Chassis twist
        Vb = np.dot(F,u) 

        # Twist Components
        wbz = Vb[0]
        vbx = Vb[1]
        vby = Vb[2]
        # Change in cordinates relative to the body
        if wbz < 0.0000000001: # If wbz is basically zero
            dqb = np.array([0,vbx,vby])
        else:
            dqb = np.array([wbz,
                            (vbx*np.sin(wbz)+vby*(np.cos(wbz)-1))/wbz,
                            (vby*np.sin(wbz)+vbx*(1-np.cos(wbz)))/wbz])

        # Configuration of space to body frame
        T_sb = np.array([[1,0,0],
                        [0,np.cos(q[0]),-1*np.sin(q[0])],
                        [0,np.sin(q[0]),np.cos(q[0])]])

        # Change from body to space frame
        dq = np.dot(T_sb,dqb)

        # New chassis configuration q
        chasis_config_next = chasis_config  + dq * timestep   #timestep ?

        gripper_state = np.array([0]) #open
        next_state = np.concatenate((chasis_config_next, joint_angle_next, wheel_angle_next, gripper_state))
        return next_state
    
    def save_data(self, data):
        '''    
        Saves the trajectory caluculated data into a csv file, so that can be imported as input to 
        copellia simulator.
        '''
        data_log = np.array(data)
        print("Generating animation csv file")
        np.savetxt('simulation_dataCA2.csv', data_log, delimiter=',', fmt='%.6f')

    

def main():
    motion = Motion()
    Traj = []

    # Setting initial values
    control_param = np.array([2,2,2,2,2,-10,10,-10,10])
    current_config = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

    T = 1 #in secs
    t = 0.01
    N = T / t
    for i in range(int(N)):     
        current_config = motion.NextState(current_config=current_config, control_param = control_param, timestep=t, max_angular_speed = 100)
        gripper_state = 0
        traj_instant = current_config + gripper_state
        Traj.append(traj_instant)

    motion.save_data(Traj)


main()
