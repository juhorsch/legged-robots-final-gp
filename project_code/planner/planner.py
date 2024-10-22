import numpy as np
from ..state import State
from .foot_planner import FootPlanner
from .base_planner import BasePlanner

class Planner:
    def __init__(self, 
                 gait : int, 
                 period : float,
                 des_linear_velocity : float,
                 des_angular_velocity : float,
                 des_acceleration : float):
        """
        Initializes the main planner object, which handles both the body and foot planners.
        Parameters:
        - gait (int): The desired gait for the legged robot.
        - period (float): The period of the gait.
        - des_linear_velocity (float): The desired linear velocity of the robot.
        - des_angular_velocity (float): The desired angular velocity of the robot.
        - des_acceleration (float): The desired acceleration of the robot.
        """
        
        self.base_planner = BasePlanner(des_linear_velocity, des_angular_velocity, des_acceleration)
        self.foot_planner = FootPlanner(gait, period, self.base_planner)
        
    def update_planner(self, state : State, **kwargs):
        """
        Updates the planner by calling the update method of the body planner and foot planner.

        Parameters:
            state (object): The state object containing the current state information.
            **kwargs: Additional keyword arguments to be passed to the update methods of the body planner and foot planner.
        """
        self.base_planner.update(state, **kwargs)
        self.foot_planner.update(state, **kwargs)

    def get_control_inputs(self, time : float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the control inputs for the given time.

        Parameters:
            time (float): The time at which to evaluate the control inputs.

        Returns:
            control_inputs (list): The control inputs for the given time.
        """
        return self.foot_planner.evaluate_trajectory(time, for_control=True)


    def evaluate_trajectory(self, time) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the trajectory at a given time.
        Parameters:
        - time: The time at which to evaluate the trajectory.
        Returns:
        - X: The concatenated array of body positions and foot positions at the given time.
        - V: The concatenated array of body velocities and foot velocities at the given time.
        - A: The concatenated array of foot accelerations at the given time.
        """

        o_X_b, o_V_b = self.base_planner.evaluate_trajectory(time)
        o_pos_feet, o_vel_feet, o_acc_feet = self.foot_planner.evaluate_trajectory(time)
        
        X = np.hstack([o_X_b.flatten(), o_pos_feet.flatten()])
        V = np.hstack([o_V_b.flatten(), o_vel_feet.flatten()])
        A = np.hstack([o_acc_feet.flatten()])
        
        return X, V, A