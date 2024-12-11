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
        self.X_error = 0.0
        self.X_error_prev = 0.0
        self.robot_state_trajectory = []

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

        # Params to control the bot (bot-states): phi, x, y, Joint1, Joint2, Joint3, Joint4, Joint5, front left wheel angle, 
        # front right wheel angle, rear right wheel angle, rear left wheel angle, gripper stateangle.
        # where J1 to J5 are the arm joint angles and W1 to W4 are the four wheel angles.

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
    # T_we_curr_actual input from the next state function.    
    def Feedback_Control(self, T_we_curr_actual, T_we_curr_ref, T_we_next_ref, Kp, Ki, timestep:int = 0.01):
        '''
        Input:

        w: world frame
        e: end-effector frame
        c: cube frame
        T_we_curr_actual (12 - np array): The current actual end-effector configuration X(q,theta) (T_se/ T_we).
                                            q: chassis configuration 
                                            θ: arm configuration  
        T_we_curr_ref (12 - np array): The current end-effector reference configuration Xd(q,theta). 
        T_we_next_ref (12 - np array): The end-effector reference configuration at the next timestep in the reference trajectory, at a time Δt later.
        Kp: The proportional gain matrice Kp.
        Ki: The integral gain matrice Ki.
        timestep: Δ t between the reference trajectory configurations.
        
        Returns:
                V_e: The commanded end-effector twist V expressed in the end-effector frame {e}.

        '''
            
        self.X_error = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(T_we_curr_actual),T_we_curr_ref)))
        print(f"X error: {self.X_error}")

        # Make X_error a matrix 
        integral_error = self.X_error_prev + self.X_error * timestep

        # use of (1/timestep) : scale the total motion into timesteps as in instantaneous velocities.
        # matrix log converts the T matrix to se3 form of twist.
        feedforward_ref_twist = (1/timestep) * mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(T_we_curr_ref),T_we_next_ref)))
        
        
        # feedback control
        Xinv_Xd  = np.dot(mr.TransInv(T_we_curr_actual),T_we_curr_ref)
        feedforward_ref_twist_in_other_frame = np.dot(mr.Adjoint(Xinv_Xd),feedforward_ref_twist)

        twist_e = feedforward_ref_twist_in_other_frame + Kp * self.X_error + Ki * integral_error
        self.X_error_prev = self.X_error

        return twist_e
 
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
        print("Generating animation csv file")
        np.savetxt('simulation_data.csv', data_log, delimiter=',', fmt='%.6f')

    def plot_error(self, X_error):
        """
        Plots the elements of X_error over time or indices.
        
        Parameters:
            X_error (np array): A NumPy array representing the error vector.
        """
        if not isinstance(X_error, np.ndarray):
            raise ValueError("X_error must be a NumPy array.")
        else:
            print("Writing error plot data.")
            # Create a new figure
            plt.figure(figsize=(8, 6))            
            # Plot each element of the X_error vector
            for i in range(X_error.shape[0]):
                plt.plot(X_error[i, :], label=f"Error component {i + 1}")
            plt.title(f"Error Plot: ")
            plt.xlabel("Time")
            plt.ylabel("Error Twist Value") # Mention the units hereeeeeeee
            plt.legend()
            plt.grid(True)
            filename = 'Error_plot'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")        

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
    

def test_traj_generator():
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

def test_feedback():
    motion = Motion()

    T_we_curr_ref = np.array([
        [0, 0, 1, 0.5],
        [0, 1, 0, 0],
        [-1, 0, 0, 0.5],
        [0, 0, 0, 1]
    ])

    T_we_next_ref = np.array([
        [0, 0, 1, 0.6],
        [0, 1, 0, 0],
        [-1, 0, 0, 0.3],
        [0, 0, 0, 1]
    ])

    T_we_curr_actual = np.array([
        [0.170, 0, 0.985, 0.387],
        [0, 1, 0, 0],
        [-0.985, 0, 0.170, 0.570],
        [0, 0, 0, 1]
    ])

    Kp = np.zeros((6,))
    Ki = np.zeros((6,))

    twist_e = motion.Feedback_Control(T_we_curr_actual=T_we_curr_actual, T_we_curr_ref=T_we_curr_ref, T_we_next_ref=T_we_next_ref, Kp=Kp, Ki=Ki)
    print(f"\n the twist value to reduce error: {twist_e}")
    # J_e: 6*9  ||  J_e_pinv: 9*6
    J_e = np.array([
        [0.030, -0.030, -0.030, 0.030, -0.985, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, -1, -1, 0],
        [-0.005, 0.005, 0.005, -0.005, 0.170, 0, 0, 0, 1],
        [0.002, 0.002, 0.002, 0.002, 0, -0.240, -0.214, -0.218, 0],
        [-0.024, 0.024, 0, 0, 0.221, 0, 0, 0, 0],
        [0.012, 0.012, 0.012, 0.012, 0, -0.288, -0.135, 0, 0]
        ])

    # Computing pseudoinverse of the Jacobian.
    J_e_pinv = np.linalg.pinv(J_e)
    # Comanded speed
    cmd_vel = J_e_pinv @ twist_e
    u = cmd_vel[0:4]
    theta_dot = cmd_vel[4:10]
    
    print(f"\n The controlled wheel velocity values: {u}")
    print(f"\n The controlled joint velocities: {theta_dot}")

def test_next_state():
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

def main_loop():
    # Initial setup
    cube_init_conf = [1,0,0]
    cube_final_conf = [0,1,-np.pi/2]
    bot_actual_init_conf = [1,0,0]
    traj_matrix = []
    Kp = np.zeros((6,))
    Ki = np.zeros((6,))
    # Total time of the motion in seconds from rest to rest
    Tf = 4 

    





    
    # initial configuration of the end-effector reference trajectory
    T_we = np.array([[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0.5],
                    [0, 0, 0, 1]])   #------------------ was given for this part - see
    
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
    
    # The number of trajectory reference configurations per 0.01 seconds(1)
    k = 1

    # Main loop
    motion = Motion()
    # Todo: Manage k and timesteps at all parts well.
    ref_traj = motion.TrajectoryGenerator(T_we_init = Tse_ini, T_wc_init = Tsc_ini, T_wc_final = Tsc_fin, T_ce_grasp = Tce_grp, T_ce_standoff = Tce_sta, k=k)

    for i in range(k):
        traj_row = motion.Feedback_Control()
        motion.NextState()
        traj_matrix.append(traj_row)
    
    # the data saved needs to be T_se vals (like from traj_generator)
    motion.save_data(traj_matrix)





    # Other variants:

    # Choose an initial configuration of the youBot so that the end-effector has at least 30 degrees of 
    # orientation error and 0.2 m of position error.

    # For feedforward:
    Kp = np.zeros((6,))
    Ki = np.zeros((6,))

    # For feedback:
    # Now add a positive-definite diagonal proportional gain matrix K p while keeping the integral gains zero.
    #  You can keep the gains "small" initially so the behavior is not much different from the 
    # case of feedforward control only. As you increase the gains, can you see some corrective effect due to the 
    # proportional control? 

    # experiment with all the options.

    # Eventually you will have to design a controller so that essentially all initial error is driven 
    # to zero by the end of the first trajectory segment; otherwise, your grasp operation may fail. 

test_traj_generator()


