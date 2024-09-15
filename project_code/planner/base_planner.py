import numpy as np
from ..constants import *
from ..state.state import State


class BasePlanner:

    def __init__(
        self,
        des_linear_velocity: float,
        des_angular_velocity: float,
        des_acceleration: float,
    ):
        """
        Initializes a BasePlanner object.
        Parameters:
        - des_linear_velocity (float): The desired linear velocity.
        - des_angular_velocity (float): The desired angular velocity.
        - des_acceleration (float): The desired acceleration.
        """

        self.des_linear_velocity = des_linear_velocity
        self.des_angular_velocity = des_angular_velocity
        self.des_acceleration = des_acceleration

    def evaluate_trajectory(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the trajectory of the robot at a given time.
        Parameters:
        - time (float): The time at which to evaluate the trajectory.
        Returns:
        - o_X_b (numpy.ndarray): The position of the robot in the world frame at the given time.
        - o_V_b (numpy.ndarray): The velocity of the robot in the world frame at the given time.
        """
        # Initialize variables for position and velocity
        pos_x, pos_y, rot_yaw = 0.0, 0.0, 0.0
        vel_x, vel_y, omg_z = 0.0, 0.0, 0.0

        # Handling the case where desired linear velocity is not zero
        if self.des_linear_velocity != 0:

            # Calculate time for acceleration phase
            acc_time = abs(self.des_linear_velocity) / self.des_acceleration
            acc = np.sign(self.des_linear_velocity) * self.des_acceleration

            # Distance covered during the acceleration phase
            if time < acc_time:
                pos_arc = 0.5 * acc * time**2
                velocity = acc * time
            else:
                # After acceleration phase, distance covered is sum of accelerating and constant velocity phases
                acc_distance = 0.5 * acc * acc_time**2

                constant_time = time - acc_time

                pos_arc = acc_distance + constant_time * self.des_linear_velocity
                velocity = self.des_linear_velocity

            # Straight line case (angular velocity is zero)
            if self.des_angular_velocity == 0:
                pos_x = pos_arc
                vel_x = velocity

            # Circular path case (angular velocity is non-zero)
            else:
                r = (
                    self.des_linear_velocity / self.des_angular_velocity
                )  # Radius of the circle

                rot_yaw = pos_arc / r  # Update yaw based on arc length
                pos_x = r * np.sin(rot_yaw)  # Compute new x position
                pos_y = r * (1 - np.cos(rot_yaw))  # Compute new y position

                vel_x = velocity * np.cos(rot_yaw)  # Tangential velocity in x-direction
                vel_y = velocity * np.sin(rot_yaw)  # Tangential velocity in y-direction
                omg_z = self.des_angular_velocity * velocity / self.des_linear_velocity  # Angular velocity

        # Position (X, Y, Z, Roll, Pitch, Yaw)
        o_X_b = np.array([[pos_x, pos_y, params["h"]], [0, 0, rot_yaw]])

        # Velocity (X_dot, Y_dot, Z_dot, Roll_dot, Pitch_dot, Yaw_dot)
        o_V_b = np.array([[vel_x, vel_y, 0], [0, 0, omg_z]])

        return o_X_b, o_V_b

    def update(self, state: State, **kwargs):
        """
        Update the base planner with the current state.

        Parameters:
        - state (State): The current state of the robot.
        - **kwargs: Additional keyword arguments.

        Returns:
        - None
        """
        pass
