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

    def TrajectoryGenerator(self, T_we_init, T_wc_init, T_wc_final, T_ce_grasp, T_ce_standoff, k:int =1):
        '''
        w: world frame
        e: end-effector frame
        c: cube frame
        T_we_init = The initial configuration of the end-effector in the reference trajectory.
        T_wc_init = The cube's initial configuration.
        T_wc_final = The cube's desired final configuration.
        T_ce_grasp = The end-effector's configuration relative to the cube when it is grasping the cube.
        T_ce_standoff = The end-effector's standoff configuration above the cube, before and after grasping, relative to the cube.
        k = The number of trajectory reference configurations per 0.01 seconds.
        Although your final animation will be based on snapshots separated by 0.01 seconds in time, the points of reference trajectory 
        (and your controller servo cycle) can be at a higher frequency. For example, if you want your controller to operate at 1000 Hz, you 
        should choose k = 10 (10 reference configurations, and 10 feedback servo cycles, per 0.01 seconds). 


        Returns (np array of shape N*13): r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state 
                Each row representing that instant of robot state and N such rows would be returned.
                where,
                T = np.array([[r11, r12, r13, px],
                  [r21, r22, r23, py],
                  [r31, r32, r33, pz],
                  [0, 0, 0, 1]])
                and N : no of iterations of config values
        '''
        gripper_state = 0 #open
        Traj = []

        # Trajectory segment 1: initial configuration to standoff configuration: a few cms above the block.
        Tf = 4 #time taken for this segment.
        N = 100*k*Tf    # if k traj in 0.01 sec, 100k traj in 1 sec, 100k*Tf traj in Tf secs.
        N = 100
        gripper_state = 0       
        T_we_init_standoff = np.dot(T_wc_init,T_ce_standoff)
        T_we_trajectory1 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_init, Xend=T_we_init_standoff, Tf=Tf, N=N, method=3))
        Traj_1 = self.convert_to_list(T_we_trajectory1,N,gripper_state)       

        # Trajectory segment 2: move the gripper down to the grasp position.
        Tf = 4 #time taken for this segment.
        N = 100
        gripper_state = 0       
        T_we_init_grasp = np.dot(T_wc_init,T_ce_grasp)
        T_we_trajectory2 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_init_standoff, Xend=T_we_init_grasp, Tf=Tf, N=N, method=3))
        Traj_2 = self.convert_to_list(T_we_trajectory2,N,gripper_state)
    
        # Trajectory segment 3: closing of the gripper.
        Tf = 4 #time taken for this segment.
        N = 100
        gripper_state = 1      
        T_we_trajectory3 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_init_grasp, Xend=T_we_init_grasp, Tf=Tf, N=N, method=3))
        Traj_3 = self.convert_to_list(T_we_trajectory3,N,gripper_state)


        # Trajectory segment 4: move the gripper back up to the standoff configuration.
        Tf = 4 #time taken for this segment.
        N = 100
        gripper_state = 1      
        T_we_trajectory4 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_init_grasp, Xend=T_we_init_standoff, Tf=Tf, N=N, method=3))
        Traj_4 = self.convert_to_list(T_we_trajectory4,N,gripper_state)

        # Trajectory segment 5: move the gripper to a standoff configuration above the final configuration.
        Tf = 4 #time taken for this segment.
        N = 100
        gripper_state = 1   
        T_we_final_standoff = np.dot(T_wc_final,T_ce_standoff)
        T_we_trajectory5 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_init_standoff, Xend=T_we_final_standoff, Tf=Tf, N=N, method=3))
        Traj_5 = self.convert_to_list(T_we_trajectory5,N,gripper_state)


        # Trajectory segment 6: move the gripper to the final configuration of the object.
        Tf = 4 #time taken for this segment.
        N = 100
        gripper_state = 1   
        T_we_final_grasp = np.dot(T_wc_final,T_ce_grasp)
        T_we_trajectory6 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_final_standoff, Xend=T_we_final_grasp, Tf=Tf, N=N, method=3))
        Traj_6 = self.convert_to_list(T_we_trajectory6,N,gripper_state)

        # Trajectory segment 7: opening of the gripper.
        Tf = 4 #time taken for this segment.
        N = 100
        gripper_state = 0
        T_we_trajectory7 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_final_grasp, Xend=T_we_final_grasp, Tf=Tf, N=N, method=3))
        Traj_7 = self.convert_to_list(T_we_trajectory7,N,gripper_state)

        # Trajectory segment 8: move the gripper back to the standoff configuration.
        Tf = 4 #time taken for this segment.
        N = 100
        gripper_state = 0  
        T_we_trajectory8 = np.asarray(mr.ScrewTrajectory(Xstart=T_we_final_grasp, Xend=T_we_final_standoff, Tf=Tf, N=N, method=3))
        Traj_8 = self.convert_to_list(T_we_trajectory8,N,gripper_state)


        Traj = Traj_1 + Traj_2 + Traj_3 + Traj_4 + Traj_5 + Traj_6 + Traj_7 + Traj_8
        self.save_data(Traj)
        return Traj
    
    def convert_to_list(self,traj,N,gripper_state):
        '''
        Helper function to convert the transformation matrix and gripper state into N x 13 matrices list  needed in csv
        '''
        list = np.zeros((N,13), dtype=float)
        Traj_result = []
        for i in range(N):
            list[i][0] = traj[i][0][0]
            list[i][1] = traj[i][0][1]
            list[i][2] = traj[i][0][2]
            list[i][3] = traj[i][1][0]
            list[i][4] = traj[i][1][1]
            list[i][5] = traj[i][1][2]
            list[i][6] = traj[i][2][0]
            list[i][7] = traj[i][2][1]
            list[i][8] = traj[i][2][2]
            list[i][9] = traj[i][0][3]
            list[i][10] = traj[i][1][3]
            list[i][11] = traj[i][2][3]
            list[i][12] = gripper_state
            Traj_result.append(list[i]) 
        return Traj_result
    

    def plan(init_cube_conf, desired_cube_conf, init_bot_conf, T_se, feedback_gains):
        csv =0
        # a data file containing the 6-vector end-effector error (the twist that would take the end-effector 
        # to the reference end-effector configuration in unit time) as a function of time
        return csv

    def save_data(self, data):
        '''    
        Saves the trajectory caluculated data into a csv file, so that can be imported as input to 
        copellia simulator.
        '''
        data_log = np.array(data)
        print("here")
        np.savetxt('simulation_data.csv', data_log, delimiter=',', fmt='%.6f')


def main():
    motion = Motion()

    Tsb = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.0963],
                    [0, 0, 0, 1]])

    Tb0 = np.array([[1, 0, 0, 0.1662],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.0026],
                    [0, 0, 0, 1]])
    M0e = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])

    Tse_ini = Tsb.dot(Tb0).dot(M0e)

    Tsc_ini = np.array([[1, 0, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.025],
                        [0, 0, 0, 1]])
    Tsc_fin = np.array([[0, 1, 0, 0],
                        [-1, 0, 0, -1],
                        [0, 0, 1, 0.025],
                        [0, 0, 0, 1]]) 

    Tce_grp = np.array([[ -np.sqrt(2)/2, 0, np.sqrt(2)/2, 0],
                        [ 0, 1, 0, 0],
                        [-np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0],
                        [ 0, 0, 0, 1]])

    Tce_sta = np.array([[ -np.sqrt(2)/2, 0, np.sqrt(2)/2,   0],
                        [ 0, 1, 0,   0],
                        [-np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0.15],  
                        [ 0, 0, 0,   1]])
    k = 1

    motion.TrajectoryGenerator(T_we_init = Tse_ini, T_wc_init = Tsc_ini, T_wc_final = Tsc_fin, T_ce_grasp = Tce_grp, T_ce_standoff = Tce_sta, k=k)



main()


# Params to control the bot (bot-states): phi, x, y, Joint1, Joint2, Joint3, Joint4, Joint5, front left wheel angle, 
# front right wheel angle, rear right wheel angle, rear left wheel angle, gripper stateangle
# where J1 to J5 are the arm joint angles and W1 to W4 are the four wheel angles