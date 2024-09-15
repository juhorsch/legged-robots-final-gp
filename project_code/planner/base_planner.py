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

