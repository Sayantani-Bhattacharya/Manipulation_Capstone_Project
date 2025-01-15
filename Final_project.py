import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt


'''
Instruction for usage: User can simply run the code and the program will generate simulation_data.csv file,
which can be used for simulation of youbot in CopelliaSim software, scene number 6.
''' 

class Motion():

    def __init__(self):
        self.frequency = 0.01
        self.X_error = []
        self.robot_state_trajectory = []
        self.integral = np.zeros((6,),dtype = float)
        self.Final_traj_matrix = []

    def matrix2list(self, whole_traj, endeffector_traj, N, gripper_state):
        '''
        This function convert the transformation matrix to a N * 13 entries list.
        '''
        sub = np.zeros((N,13),dtype = float)
        for i in range(N):
            sub[i][0] = endeffector_traj[i][0][0]
            sub[i][1] = endeffector_traj[i][0][1]
            sub[i][2] = endeffector_traj[i][0][2]
            sub[i][3] = endeffector_traj[i][1][0]
            sub[i][4] = endeffector_traj[i][1][1]
            sub[i][5] = endeffector_traj[i][1][2]
            sub[i][6] = endeffector_traj[i][2][0]
            sub[i][7] = endeffector_traj[i][2][1]
            sub[i][8] = endeffector_traj[i][2][2]
            sub[i][9] = endeffector_traj[i][0][3]
            sub[i][10] = endeffector_traj[i][1][3]
            sub[i][11] = endeffector_traj[i][2][3]
            sub[i][12] = gripper_state
            whole_traj.append(sub[i].tolist())	
        return whole_traj

    def NextState(self, curr_config,control_param,time_step, max_angular_speed):
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

        # set variables
        r = 0.0475
        l = 0.235
        w = 0.15
        # get current configuration from input,qk is the configuration of chasis, jangle for joint angles and wangle for wheel angles
        qk = np.array([[curr_config[0]],[curr_config[1]],[curr_config[2]]])
        curr_jangle = np.array([[curr_config[3]],[curr_config[4]],[curr_config[5]],[curr_config[6]],[curr_config[7]]])
        curr_wangle = np.array([[curr_config[8]],[curr_config[9]],[curr_config[10]],[curr_config[11]]])
        # derive next joint configuration
        theta_dot = np.array([[control_param[0]],[control_param[1]],[control_param[2]],[control_param[3]],[control_param[4]]])
        det_theta = theta_dot*time_step
        next_jangle = curr_jangle + det_theta      
        
        # derive next wheel configuration
        u = np.array([[control_param[5]],[control_param[6]],[control_param[7]],[control_param[8]]])
        det_u = u*time_step
        next_wangle = curr_wangle + det_u
        # compute next chasis configuration
        F = r/4 * np.array([[-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],[1,1,1,1],[-1,1,-1,1]])
        V_b = np.dot(F,det_u).reshape(3,)
        wbz = V_b[0]
        vbx = V_b[1]
        vby = V_b[2]
        det_qb = np.zeros(3,)

        if wbz < 1e-3:
            det_qb = np.array([[0],[vbx],[vby]])
        else:
            det_qb = np.array([[wbz],[vbx*np.sin(wbz)+vby*(np.cos(wbz)-1)/wbz],[vby*np.sin(wbz)+vbx*(1-np.cos(wbz))/wbz]])          

        update_matrix = np.array([[1,0,0],
                                [0,np.cos(curr_config[0]),-np.sin(curr_config[0])],
                                [0,np.sin(curr_config[0]),np.cos(curr_config[0])]])

        det_q = np.dot(update_matrix,det_qb)
        next_q = qk + det_q
        # output the next configuration
        next_config = np.concatenate((next_q, next_jangle,next_wangle), axis=None)
        return next_config

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
            # self.save_data(Traj)
            return Traj

    def convert_to_list(self, traj,N,gripper_state):
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
        
    def save_data(self, data):
        '''    
        Saves the trajectory caluculated data into a csv file, so that can be imported as input to 
        copellia simulator.
        '''
        data_log = np.array(data)
        print("Generating animation csv file")
        np.savetxt('simulation_data.csv', data_log, delimiter=',', fmt='%.6f')

    def FeedbackControl(self, integral,robot_config,T_we_curr_actual,T_we_curr_ref,T_we_next_ref,Kp,Ki,time_step):                
        """
        Calculates the kinematic task-space feedforward plus feedback control law.

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
                    V_e: The (tobe) commanded end-effector twist V expressed in the end-effector frame {e}.
                    X_err: The transform error between T_we_curr_actual and T_we_curr_ref.
    
        """
        r = 0.0475
        l = 0.235
        w = 0.15
        Tb0 = np.array([[ 1, 0, 0, 0.1662],
                        [ 0, 1, 0,   0],
                        [ 0, 0, 1, 0.0026],
                        [ 0, 0, 0,   1]])

        M0e = np.array([[ 1, 0, 0, 0.033],
                        [ 0, 1, 0,   0],
                        [ 0, 0, 1, 0.6546],
                        [ 0, 0, 0,   1]])
        
        # Blist: The joint screw axes in the end-effector frame when the manipulator is at the home position.                  		
        Blist = np.array([[0, 0, 1,   0, 0.033, 0],
                        [0,-1, 0,-0.5076,  0, 0],
                        [0,-1, 0,-0.3526,  0, 0],
                        [0,-1, 0,-0.2176,  0, 0],
                        [0, 0, 1,   0,     0, 0]]).T
        
        X = T_we_curr_actual
        Xd = T_we_curr_ref
        Xd_next = T_we_next_ref

        # thetalist: A list of joint coordinates
        thetalist = robot_config[3:8]
        F = r/4 * np.array([[0,0,0,0],[0,0,0,0],[-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],[1,1,1,1],[-1,1,-1,1],[0,0,0,0]])
        T0e = mr.FKinBody(M0e,Blist,thetalist)
        Vd =  mr.se3ToVec((1/time_step)*mr.MatrixLog6(np.dot(mr.TransInv(Xd),Xd_next)))
        ADxxd = mr.Adjoint(np.dot(mr.TransInv(X),Xd))
        ADxxdVd = np.dot(ADxxd,Vd)
        Xerr = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(X),Xd)))
        V = ADxxdVd + np.dot(Kp,Xerr) +  np.dot(Ki,integral + Xerr * time_step)
        integral +=  Xerr * time_step
        Ja = mr.JacobianBody(Blist, thetalist)
        Jb = mr.Adjoint(np.dot(mr.TransInv(T0e),mr.TransInv(Tb0))).dot(F)
        Je = np.concatenate((Jb,Ja),axis=1)
        Je_inv = np.linalg.pinv(Je)
        command = Je_inv.dot(V)
        return command,Xerr

    def main_loop(self, Tsc_ini,Tsc_fin,KP,KI,robot_config):
        """
        This function implements a basic position controlled motion for a wheeled mobile 
        robot.

        Main function call, that generates the animation csv and plot for the transformation 
        error between the current and desired reference positions.        

        Input:  
            cube_init_conf = Tsc_ini: The cube's initial configruation relative to the ground
            cube_final_conf = Tsc_fin: The cube's final configruation relative to the ground
            Kp: P controller gain
            Ki: I controller gain
            robot_config: The initial configuration of the youBot

        Output: 
            csv file that has: np array of the state to get the youBot desired trajectory. 
            The file is to be used as as input in copelliaSim to observe the desired motion.
            
        """
        bot_actual_init_conf = robot_config
        # set up the initial variables
        Tse_ini = np.array([[ 0, 0, 1,   0],
                        [ 0, 1, 0,   0],
                        [ -1, 0,0, 0.5],
                        [ 0, 0, 0,   1]])


        Tce_grp = np.array([[ -np.sqrt(2)/2, 0, np.sqrt(2)/2, 0],
                        [ 0, 1, 0, 0],
                        [-np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0],
                        [ 0, 0, 0, 1]])

        Tce_sta = np.array([[ -np.sqrt(2)/2, 0, np.sqrt(2)/2, 0],
                        [ 0, 1, 0, 0],
                        [-np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0.1],
                        [ 0, 0, 0, 1]])
        Tb0 = np.array([[ 1, 0, 0, 0.1662],
                        [ 0, 1, 0,   0],
                        [ 0, 0, 1, 0.0026],
                        [ 0, 0, 0,   1]])
        Blist = np.array([[0, 0, 1,   0, 0.033, 0],
                        [0,-1, 0,-0.5076,  0, 0],
                        [0,-1, 0,-0.3526,  0, 0],
                        [0,-1, 0,-0.2176,  0, 0],
                        [0, 0, 1,   0,     0, 0]]).T
        M0e = np.array([[ 1, 0, 0, 0.033],
                        [ 0, 1, 0,   0],
                        [ 0, 0, 1, 0.6546],
                        [ 0, 0, 0,   1]])
        k = 1
        speed_max = 10
        time_step = 0.01

        traj = np.asarray(self.TrajectoryGenerator(Tse_ini, Tsc_ini, Tsc_fin, Tce_grp, Tce_sta, k))
        # append the initial configuration to the whole trajectory
        self.Final_traj_matrix.append(robot_config.tolist())
        # begin the loop
        for i in range (len(traj)-1):
            # joint angle
            # every time update variables 
            thetalist = robot_config[3:8]		
            Xd = np.array([[ traj[i][0], traj[i][1], traj[i][2],  traj[i][9]],
                        [ traj[i][3], traj[i][4], traj[i][5], traj[i][10]], 
                        [ traj[i][6], traj[i][7], traj[i][8], traj[i][11]],
                        [          0,          0,          0,          1]])
            Xd_next = np.array([[ traj[i+1][0], traj[i+1][1], traj[i+1][2],  traj[i+1][9]],
                                [ traj[i+1][3], traj[i+1][4], traj[i+1][5], traj[i+1][10]],
                                [ traj[i+1][6], traj[i+1][7], traj[i+1][8], traj[i+1][11]],
                                [            0,            0,            0,           1]])
            Tsb = np.array([[np.cos(robot_config[0]),-np.sin(robot_config[0]), 0, robot_config[1]],
                            [np.sin(robot_config[0]), np.cos(robot_config[0]), 0, robot_config[2]], 
                            [              0       ,          0            , 1,        0.0963 ],
                            [              0       ,          0            , 0,             1]])
            T0e = mr.FKinBody(M0e,Blist,thetalist)
            X = np.dot(Tsb,np.dot(Tb0,T0e))
            # get the command and error vector from feedback control
            command,Xerr = self.FeedbackControl(self.integral,robot_config,X,Xd,Xd_next,KP,KI,time_step)
            self.X_error.append(Xerr.tolist())
            Cw = command[:4]
            Cj = command[4:9]
            # the input command of NextState and the command returned by FeedbackControl is flipped
            controls = np.concatenate((Cj,Cw),axis=None)
            robot_config = self.NextState(robot_config[:12],controls,time_step=0.01,max_angular_speed=10)
            traj_instant = np.concatenate((robot_config,traj[i][12]),axis=None)
            self.Final_traj_matrix.append(traj_instant.tolist())

        self.plot_error(self.X_error)
        self.save_animation_csv(self.Final_traj_matrix)

    def plot_error(self, X_error):
        '''
        Helper function to plot the elements of X_error over time or indices.
        Also save the log data in Xerrnewtask.csv.

        Input:
            X_error (np array): A NumPy array representing the error vector.
        '''

        # save the Xerr vector
        print('generating Xerr data file')
        np.savetxt('Xerrnewtask.csv', X_error, delimiter=',')
        # plot the Xerr
        print('plotting error data')
        qvec = np.asarray(X_error)
        tvec = np.linspace(0,13.99,799) 
        plt.plot(tvec,qvec[:,0])
        plt.plot(tvec,qvec[:,1])
        plt.plot(tvec,qvec[:,2])
        plt.plot(tvec,qvec[:,3])
        plt.plot(tvec,qvec[:,4])
        plt.plot(tvec,qvec[:,5])
        plt.xlim([0,14])
        plt.title(' Xerr plot')
        plt.xlabel('Time (s)')
        plt.ylabel('error')
        plt.legend([r'$Xerr[1]$',r'$Xerr[2]$',r'$Xerr[3]$',r'$Xerr[4]$',r'$Xerr[5]$',r'$Xerr[6]$'])
        plt.grid(True)
        # plt.show()
        plt.savefig("Error_plot.png")

    def save_animation_csv(self, final_traj_matrix):
        '''
        Helper function to save a csv from the trajectory data of main loop.
        '''
        print("Generating animation csv file")
        np.savetxt('simulation_data.csv', final_traj_matrix, delimiter=',')
        print('Done')


# Testing functions
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
    integral = np.zeros((6,),dtype = float)
    robot_config = np.array([0.1,0.1,0.2,0,0,0.2,-1.6, 0,0,0,0,0,0])
    # cmd_vel = motion.FeedbackControl(integral,robot_config,X,Xd,Xd_next,KP,KI,time_step):
    cmd_vel = motion.FeedbackControl(T_we_curr_actual=T_we_curr_actual, T_we_curr_ref=T_we_curr_ref, T_we_next_ref=T_we_next_ref, Kp=Kp, Ki=Ki)
    
    u = cmd_vel[0:4]
    theta_dot = cmd_vel[4:10]
    print(f"\n The controlled wheel velocity values: {u}")
    print(f"\n The controlled joint velocities: {theta_dot}")

def test_main_loop():
    kp = 100
    ki = 30
    Tsc_ini = np.array([[1, 0, 0,     1.2],
                        [0, 1, 0,     0],
                        [0, 0, 1, 0],
                        [0, 0, 0,     1]])

    Tsc_fin = np.array([[ 0, 1, 0,     0],
                        [-1, 0, 0,    -1.7],
                        [ 0, 0, 1, 0],
                        [ 0, 0, 0,     1]])
    KP = np.array([[kp, 0, 0, 0, 0, 0],
                [ 0,kp, 0, 0, 0, 0],
                [ 0, 0,kp, 0, 0, 0],
                [ 0, 0, 0,kp, 0, 0],
                [ 0, 0, 0, 0,kp, 0],
                [ 0, 0, 0, 0, 0,kp]])
    KI = np.array([[ki, 0, 0, 0, 0, 0],
                [ 0,ki, 0, 0, 0, 0],
                [ 0, 0,ki, 0, 0, 0],
                [ 0, 0, 0,ki, 0, 0],
                [ 0, 0, 0, 0,ki, 0],
                [ 0, 0, 0, 0, 0,ki]])
    robot_config = np.array([0.1,0.1,0.2,0,0,0.2,-1.6, 0,0,0,0,0,0])
    motion = Motion()
    motion.main_loop(Tsc_ini,Tsc_fin,KP,KI,robot_config)

# To generate the csv and plots.
test_main_loop()
