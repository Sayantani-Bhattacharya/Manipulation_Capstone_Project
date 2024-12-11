import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
import math
import time

class Motion():

    def __init__(self):
        self.frequency = 0.01
        self.X_error = 0.0
        self.X_error_prev = 0.0


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
    

# Testing func
def main():
    motion = Motion()

    # robot configuration: ( ϕ , x , y , θ 1 , θ 2 , θ 3 , θ 4 , θ 5 ) = ( 0 , 0 , 0 , 0 , 0 , 0.2 , − 1.6 , 0 )
    # Define the matrices as NumPy arrays
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





main()