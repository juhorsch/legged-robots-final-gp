import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

from ..constants import *
from ..state.state import State
from .mpc import MPC
from .base_planner import BasePlanner
from ..utils.linearize_dynamics import get_AB_matrices, get_inequality_constraints


def evaluate_cubic_polynomial(pol_coef: np.array, t: float) -> float:
    """
    Evaluates a cubic polynomial at a given time point.

    Parameters:
    pol_coef (ndarray): Coefficients of the cubic polynomial.
    t (float): Time point at which to evaluate the polynomial.

    Returns:
    float: The value of the cubic polynomial at the given time point.
    """
    a, b, c, d = (
        pol_coef[..., 0],
        pol_coef[..., 1],
        pol_coef[..., 2],
        pol_coef[..., 3],
    )
    return a * t**3 + b * t**2 + c * t + d


class FootPlanner:

    def __init__(self, gait: int, period: float, base_planner: BasePlanner):
        """
        Initializes the FootPlanner object.

        Parameters:
        - gait (int): The gait type.
        - period (float): The period of the gait.
        - base_planner (BasePlanner): The base planner object.

        Returns:
        None
        """

        self.gait = gait
        self.period = period
        self.base_planner = base_planner

        self.timestamp = None
        self.pol_coef = None
        self.sw_prog = None
        self.forces = None

        self.t_ab, self.t_st, self.t_sw = self._create_t_ab(gait, period)
        self.o_pos_sim_point = np.ones((4, 3))
        self.o_pos_feet_stance = np.zeros((4, 3))

    def evaluate_trajectory(
        self, time, for_control=False
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """
        Evaluates the trajectory of the legged robot at a given time.
        Parameters:
        - time (float): The time at which to evaluate the trajectory.
        - for_control (bool): Flag indicating whether the trajectory is being evaluated for control purposes.
        Returns:
        - If for_control is True:
            - o_pos_feet (numpy.array): Array containing foot positions at the given time in world frame.
            - o_vel_feet (numpy.array): Array containing foot velocities at the given time in world frame.
            - o_acc_feet (numpy.array): Array containing foot accelerations at the given time in world frame.
            - o_forces (numpy.array): Array containing forces at the given time in world frame.
            - stance_state (numpy.array): The stance state for each foot at the given time.
        - If for_control is False:
            - o_pos_feet (numpy.array): Array containing foot positions at the given time in world frame.
            - o_vel_feet (numpy.array): Array containing foot velocities at the given time in world frame.
            - o_acc_feet (numpy.array): Array containing foot accelerations at the given time in world frame.
        """
        assert (
            self.timestamp is not None
        ), "Please call update() before calling evaluate_trajectory()!"

        delta_time = time - self.timestamp
        sw_prog = self._calculate_swing_progress_trajectory(delta_time, 1)

        o_pos_feet = self._evaluate_foot_trajectory(
            type=POSITION, pol_coef=self.pol_coef, sw_prog=sw_prog
        )
        o_vel_feet = self._evaluate_foot_trajectory(
            type=VELOCITY, pol_coef=self.pol_coef, sw_prog=sw_prog
        )
        o_acc_feet = self._evaluate_foot_trajectory(
            type=ACCELERATION, pol_coef=self.pol_coef, sw_prog=sw_prog
        )

        if for_control:
            o_forces = self._evaluate_forces(time)
            stance_state = self._convert_to_stance_state(sw_prog)
            return o_pos_feet, o_vel_feet, o_acc_feet, o_forces, stance_state

        else:
            return o_pos_feet, o_vel_feet, o_acc_feet

    def update(self, state: State, **kwargs) -> None:
        """
        Updates the foot planner based on the current state.
        Parameters:
            state (State): The current state of the robot.
            **kwargs: Additional keyword arguments.
        Returns:
            None
        """
        self.timestamp = state.time()
        self._update_swing_progress()
        self._update_stance_position(state)
        self._update_polynomial_coefficients(state)

        if not (kwargs.get("mode") == NO_FORCE):
            self._update_forces(state)

    def _evaluate_forces(self, time: float) -> np.ndarray:
        """
        Evaluate the forces acting on the robot's legs at a given time.

        Parameters:
            time (float): The time at which the forces are evaluated.

        Returns:
            numpy.ndarray: A 2D numpy array of shape (NUM_LEGS, 3) representing the forces acting on each leg.
        """
        return np.array([f(time) for f in self.forces]).reshape((NUM_LEGS, 3))

    @staticmethod
    def _convert_to_stance_state(sw_prog: np.ndarray) -> np.ndarray:
        """
        Converts the given swing phase progress to a stance state.

        Parameters:
        - sw_prog (numpy.ndarray) : The swing phase progress.

        Returns:
        - bool: True if the swing phase progress is 0 or 1, False otherwise.
        """
        return (sw_prog <= 0) | (sw_prog >= 1)

    def _update_swing_progress(self) -> None:
        """
        Updates the swing progress of the foot planner.

        This method calculates the swing progress of the foot planner based on the current timestamp,
        the swing period, and the swing start time. It uses the formula:

        sw_prog = 1 / t_sw * (t_dash - t_st)

        where:
        - sw_prog is the swing progress
        - t_sw is the swing period
        - t_dash is the current timestamp modulo the swing period
        - t_st is the swing start time

        Returns:
        None
        """
        t_dash = np.clip(
            np.fmod(self.timestamp - self.t_ab[:, 0], self.period), 0, self.period
        )
        self.sw_prog = 1 / (self.t_sw) * (t_dash - self.t_st)

    def _update_stance_position(self, state: State) -> None:
        """
        Update the stance position of the feet based on the given state.

        Parameters:
        - state: The current state of the robot.

        Returns:
        - None
        """
        o_pos_feet = state.get_feet_position()
        for i in range(NUM_LEGS):
            if self.sw_prog[i] < 0:
                self.o_pos_feet_stance[i, :] = o_pos_feet[i, :]

    def _update_polynomial_coefficients(self, state: State) -> None:
        """
        Update the polynomial coefficients for position, velocity, and acceleration.

        Parameters:
            state (State): The state of the robot.

        Returns:
            None
        """
        coef_x, coef_y = self._compute_xy_coefficients(state)
        coef_z = self._compute_z_coefficients()

        coef_pos = np.concatenate([coef_x, coef_y, coef_z], axis=1)
        coef_vel = self._differentiate_coefficients(coef_pos)
        coef_acc = self._differentiate_coefficients(coef_vel)

        self.pol_coef = (coef_pos, coef_vel, coef_acc)

    def _compute_xy_coefficients(self, state) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the x and y coefficients for the swing end foot position.

        Parameters:
        - state: The current state of the robot.

        Returns:
        - coef_x: The x coefficients for the swing end foot position.
        - coef_y: The y coefficients for the swing end foot position.
        """
        o_pos_sw_end = self._get_swing_end_foot_position(state)
        o_pos_st_end = self.o_pos_feet_stance

        def create_polynomial(start, end):
            return np.stack(
                [2 * (start - end), 3 * (end - start), np.zeros_like(start), start],
                axis=-1,
            )[..., None, :]

        coef_x = create_polynomial(o_pos_st_end[:, 0], o_pos_sw_end[:, 0])
        coef_y = create_polynomial(o_pos_st_end[:, 1], o_pos_sw_end[:, 1])

        return coef_x, coef_y

    @staticmethod
    def _compute_z_coefficients() -> np.ndarray:
        """
        Compute the z coefficients for raising and lowering the foot.
        Returns:
            np.ndarray: Array of z coefficients for each leg.
        """
        h_max, h_min = params["h_max"], params["h_min"]
        delta_h = h_min - h_max

        # Combine both coef_z1 and coef_z2 before tiling
        coef_z = np.array(
            [
                [-16 * h_max, 12 * h_max, 0, h_min],  # Raising the foot
                [
                    -16 * delta_h,
                    36 * delta_h,
                    -24 * delta_h,
                    h_max + 5 * delta_h,
                ],  # Lowering the foot
            ]
        )

        # Tile both sets of coefficients for each leg in one step
        coef_z = np.tile(coef_z, (NUM_LEGS, 1, 1))

        return coef_z

    @staticmethod
    def _differentiate_coefficients(coef: np.ndarray) -> np.ndarray:
        """
        Differentiates the coefficients of a polynomial.

        Parameters:
        coef (ndarray): The coefficients of the polynomial.

        Returns:
        ndarray: The differentiated coefficients.

        """
        coef_diff = np.zeros_like(coef)
        coef_diff[..., 1] = 3 * coef[..., 0]
        coef_diff[..., 2] = 2 * coef[..., 1]
        coef_diff[..., 3] = coef[..., 2]
        return coef_diff

    @staticmethod
    def _add_gravity_to_state(state_vector: np.ndarray) -> np.ndarray:
        """
        Add gravity to the state vector.
        Parameters:
        - state_vector (np.ndarray): The input state vector.
        Returns:
        - np.ndarray: The state vector with gravity added.
        """
        gravity = params["g"]

        if len(state_vector.shape) == 1:
            state_vector = np.append(state_vector, -gravity)
        else:
            n = state_vector.shape[0]
            gravity_state = -np.ones((n, 1)) * gravity
            state_vector = np.hstack([state_vector, gravity_state])
        return state_vector

    def _update_forces(self, state: State) -> None:
        """
        Updates the forces applied to the robot's feet based on the current state.
        Parameters:
        - state (State): The current state of the robot.
        Returns:
        - None
        """
        trajectory_state, trajectory_feet, trajectory_stance_state = (
            self._plan_receding_horizon()
        )

        At, Bt = get_AB_matrices(trajectory_state, trajectory_feet)
        C, l, h = get_inequality_constraints(trajectory_stance_state)

        xk = state.get_body_state()
        xk = self._add_gravity_to_state(xk)

        self.forces = self._solve_mpc(xk, trajectory_state, At, Bt, C, l, h)

    def _solve_mpc(self, xk, trajectory_state, At, Bt, C, l, h) -> list:
        """
        Solves the Model Predictive Control (MPC) problem for a given state and trajectory.
        Parameters:
        - xk: Current state of the system.
        - trajectory_state: Desired trajectory state.
        - At: State transition matrix.
        - Bt: Control input matrix.
        - C: External disturbance matrix.
        - l: Lower bound on control input.
        - h: Upper bound on control input.
        Returns:
        - forces: Control forces calculated by the MPC controller.
        """
        # Extract parameters
        n_pred = params["pred_horizon"]
        Q, R, QT = params["Q"], params["R"], params["QT"]

        controller = MPC(
            xk=xk,
            Xd=trajectory_state,
            N=n_pred,
            Ak=At,
            Bk=Bt[0],
            C_ext=C,
            l_ext=l,
            h_ext=h,
            Q=Q,
            R=R,
            QT=QT,
            verbose=False,
            measure_runtime=True,
        )
        force_trajectory = controller.solve()
        forces = self._interpolate_forces(force_trajectory)

        return forces

    def _interpolate_forces(self, force_trajectory: np.ndarray) -> list:
        """
        Interpolates the force trajectory over a given time horizon.
        Parameters:
        - force_trajectory (numpy.ndarray): The force trajectory to be interpolated.
        Returns:
        - list: A list of interpolated forces for each leg.
        """
        # Extract parameters
        n_pred, t_mpc = params["pred_horizon"], params["t_mpc"]

        time_steps = np.linspace(
            self.timestamp, self.timestamp + (n_pred - 1) * t_mpc, n_pred
        )
        forces = [
            interp1d(time_steps, force_trajectory[:, i], kind="linear")
            for i in range(12)
        ]
        return forces

    def _get_swing_end_foot_position(self, state: State) -> np.ndarray:
        """
        Calculates the foot position at the end of the swing phase based on the robot's state.
        Parameters:
            state (State): The current state of the robot.
        Returns:
            np.ndarray: A 4x3 array representing the swing end foot positions for each leg.
        """
        o_pos_b = state.get_body_position()
        o_rot_b = state.get_body_rotation_euler()
        o_vel_b = state.get_body_velocity()
        o_omega_b = state.get_body_angular_velocity()
        o_vel_b_des = self.base_planner.evaluate_trajectory(self.timestamp)[1][0]
        
        hip_x, hip_y = self._compute_hip_positions()
        o_pos_sw_end = np.zeros((4, 3))

        for i in range(NUM_LEGS):
            # Compute the symmetrical point for the current leg
            o_pos_sim_point = self._compute_symmetrical_point_for_leg(i, o_pos_b, o_rot_b, o_vel_b, o_omega_b, hip_x[i], hip_y[i])

            # Compute the foot position adjustment for the current leg
            delta_p = self._compute_delta_p(i, state, o_vel_b, o_omega_b, o_vel_b_des)

            # Final foot position at swing end
            o_pos_sw_end[i] = o_pos_sim_point + delta_p

        return o_pos_sw_end


    def _compute_hip_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the x and y positions of the hips based on robot dimensions and leg configuration.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Hip x and y positions for each leg.
        """
        hip_x = params["L"] * np.array(params["sign_L"])
        hip_y = (params["W"] + params["d"]) * np.array(params["sign_d"])
        return hip_x, hip_y


    def _compute_symmetrical_point_for_leg(self, i: int, o_pos_b: np.ndarray, o_rot_b: np.ndarray,
                                        o_vel_b: np.ndarray, o_omega_b: np.ndarray, hip_x: float, hip_y: float) -> np.ndarray:
        """
        Compute the symmetrical point for a specific leg.
        Parameters:
            i (int): Leg index.
            o_pos_b (np.ndarray): Body position.
            o_rot_b (np.ndarray): Body rotation (Euler angles).
            o_vel_b (np.ndarray): Body velocity.
            o_omega_b (np.ndarray): Body angular velocity.
            hip_x (float): Hip x position for the leg.
            hip_y (float): Hip y position for the leg.
        Returns:
            np.ndarray: The symmetrical point in world coordinates.
        """
        o_pos_b_touch = o_pos_b + o_vel_b * (1 - self.sw_prog[i]) * self.t_sw[i]
        yaw_touch = o_rot_b[2] + o_omega_b[2] * (1 - self.sw_prog[i]) * self.t_sw[i]
        b_pos_hip_i = np.array([hip_x, hip_y, 0])
        
        return o_pos_b_touch + Rotation.from_euler("z", yaw_touch).as_matrix() @ b_pos_hip_i


    def _compute_delta_p(self, i: int, state: State, o_vel_b: np.ndarray, o_omega_b: np.ndarray, 
                        o_vel_b_des: np.ndarray) -> np.ndarray:
        """
        Compute the change in foot position (delta_p) for a specific leg.
        Parameters:
            i (int): Leg index.
            state (State): The current state of the robot.
            o_vel_b (np.ndarray): Body velocity.
            o_omega_b (np.ndarray): Body angular velocity.
            o_vel_b_des (np.ndarray): Desired body velocity.
        Returns:
            np.ndarray: The change in foot position for the leg.
        """
        K, gravity = params["k_fst"], params["g"]
        o_z_b = state.get_body_position()[2]
        
        dp1 = self._compute_dp1(i, o_vel_b)
        dp2 = self._compute_dp2(i, state, o_omega_b)
        dp3 = K * (o_vel_b - o_vel_b_des)
        dp4 = self._compute_dp4(gravity, o_z_b, o_vel_b, o_omega_b)

        return dp1 + dp2 + dp3 + dp4


    def _compute_dp1(self, i: int, o_vel_b: np.ndarray) -> np.ndarray:
        """
        Compute the first delta term (based on body velocity and stance time).
        Parameters:
            i (int): Leg index.
            o_vel_b (np.ndarray): Body velocity.
        Returns:
            np.ndarray: The first delta term.
        """
        return o_vel_b * self.t_st[i] * 0.5


    def _compute_dp2(self, i: int, state: State, o_omega_b: np.ndarray) -> np.ndarray:
        """
        Compute the second delta term (based on body yaw and hip position).
        Parameters:
            i (int): Leg index.
            state (State): The current state of the robot.
            o_omega_b (np.ndarray): Body angular velocity.
        Returns:
            np.ndarray: The second delta term.
        """
        o_rot_b = state.get_body_rotation_euler()
        yaw_touch = o_rot_b[2] + o_omega_b[2] * (1 - self.sw_prog[i]) * self.t_sw[i]
        hip_x, hip_y = self._compute_hip_positions()
        b_pos_hip_i = np.array([hip_x[i], hip_y[i], 0])

        return Rotation.from_euler("z", yaw_touch).as_matrix() @ (
            Rotation.from_euler("z", (o_omega_b[2] * self.t_st[i] / 2)).as_matrix() - np.eye(3)
        ) @ b_pos_hip_i


    def _compute_dp4(self, gravity: float, o_z_b: float, o_vel_b: np.ndarray, o_omega_b: np.ndarray) -> np.ndarray:
        """
        Compute the fourth delta term (based on gravity and cross-product of velocities).
        Parameters:
            gravity (float): Gravitational constant.
            o_z_b (float): Body z-position.
            o_vel_b (np.ndarray): Body velocity.
            o_omega_b (np.ndarray): Body angular velocity.
        Returns:
            np.ndarray: The fourth delta term.
        """
        if gravity == 0:
            return 0
        return o_z_b / gravity * np.cross(o_vel_b, o_omega_b)


    def _plan_receding_horizon(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plans the trajectory for a receding horizon based on the body planner and the foot stance positions.
        Returns:
            trajectory_state (ndarray): The state trajectory for the prediction horizon.
            trajectory_feet (ndarray): The feet trajectory for the prediction horizon.
            trajectory_stance_state (ndarray): The stance indicator for the prediction horizon.
        """
        # Extract parameters
        n_pred, t_mpc = params["pred_horizon"], params["t_mpc"]

        # Initialize the state trajectory - N x (Pose, Velocity) x (Position, Orientation) x (X, Y, Z)
        trajectory_state = np.zeros((n_pred, 2, 2, 3))

        # Initialize the feet trajectory
        trajectory_feet = np.zeros((n_pred, NUM_LEGS, 3))

        # Calculate the future swing progress values
        trajectory_sw_prog = self._calculate_swing_progress_trajectory(t_mpc, n_pred)

        # Loop through prediction horizon
        for k in range(n_pred):
            trajectory_state[k, 0], trajectory_state[k, 1] = (
                self.base_planner.evaluate_trajectory(self.timestamp + k * t_mpc)
            )
            trajectory_feet[k] = self._evaluate_foot_trajectory(
                type=POSITION, pol_coef=self.pol_coef, sw_prog=trajectory_sw_prog[k]
            )

        # Reshape to Dimension N x 12
        trajectory_state = np.reshape(trajectory_state, (n_pred, 12))

        # Add gravity to state
        trajectory_state = self._add_gravity_to_state(trajectory_state)

        # Convert to stance indicator - 0 in swing - 1 in stance
        trajectory_stance_state = self._convert_to_stance_state(trajectory_sw_prog)

        return trajectory_state, trajectory_feet, trajectory_stance_state

    @staticmethod
    def _create_t_ab(gait, period) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        calculates the limbs' ground contact and departure time

        :param gait: Gait identifier - currently just WALKING_TROT implemented
        :param period: cycle duration

        :return:
            t_ab: 4x2 array with start and end times of the stance phase for each limb
        """
        t_ab = np.zeros((NUM_LEGS, 2))
        if gait == WALKING_TROT:
            t_st = period * (2 / 3)

            t_ab[FRONT_LEFT, :] = [0, t_st]
            t_ab[BACK_RIGHT, :] = [0, t_st]

            t_ab[FRONT_RIGHT, 0] = 1 / 2 * period
            t_ab[BACK_LEFT, 0] = 1 / 2 * period

            t_ab[FRONT_RIGHT, 1] = t_ab[FRONT_RIGHT, 0] + t_st
            t_ab[BACK_LEFT, 1] = t_ab[BACK_LEFT, 0] + t_st
            
        elif gait == CRAWL:
            t_st = period * 0.75
            t_sw = period - t_st

            t_ab[FRONT_LEFT, 0] = 0
            t_ab[FRONT_RIGHT, 0] = t_sw
            t_ab[BACK_LEFT, 0] = t_sw * 2
            t_ab[BACK_RIGHT, 0] = t_sw * 3

            t_ab[:, 1] = t_ab[:, 0] + t_st 

        t_st = t_ab[:, 1] - t_ab[:, 0]
        t_sw = period - t_st
        return t_ab, t_st, t_sw

    @staticmethod
    def _evaluate_foot_trajectory(type, pol_coef, sw_prog) -> np.ndarray:
        """
        Evaluate the trajectory of the feet based on the given polynomial coefficients and swing progress.

        Parameters:
        - type (int): The type of trajectory. (POSITION, VELOCITY or ACCELERATION)
        - pol_coef (ndarray): The polynomial coefficients for each foot and trajectory type.
        - sw_prog (ndarray): The swing progress for each foot.

        Returns:
        - trajectory_feet (ndarray): The evaluated trajectory of the feet.

        """
        trajectory_feet = np.zeros((NUM_LEGS, 3))
        for foot in range(NUM_LEGS):
            sw_k = sw_prog[foot]
            if sw_k < 0.5:  # Foot is lifting
                trajectory_feet[foot] = evaluate_cubic_polynomial(
                    pol_coef[type][foot, :3], sw_k
                )
            else:  # Foot is lowering
                trajectory_feet[foot] = evaluate_cubic_polynomial(
                    pol_coef[type][foot, [0, 1, 3]], sw_k
                )
        return trajectory_feet

    def _calculate_swing_progress_trajectory(
        self, delta_time: float, n_steps: int
    ) -> np.ndarray:
        """
        Calculates the swing progress trajectory based on the given time step and number of steps.
        Parameters:
        - delta_time (float): The time step.
        - n_steps (int): The number of steps.
        Returns:
        - trajectory_sw_prog (numpy.ndarray): The swing progress trajectory.
        """

        if n_steps == 1:
            delta_sw_prog = delta_time / self.t_sw

        else:
            step_size = delta_time / self.t_sw
            delta_sw_prog = np.arange(n_steps)[:, None] * step_size[None, :]

        trajectory_sw_prog = self.sw_prog + delta_sw_prog
        trajectory_sw_prog = np.clip(trajectory_sw_prog, a_min=0, a_max=1)

        return trajectory_sw_prog
    
    def calculate_sw_prog(self, time):
        t_dash = np.clip(
            np.fmod(time - self.t_ab[:, 0], self.period), 0, self.period
        )
        sw_prog = 1 / (self.t_sw) * (t_dash - self.t_st)
        return sw_prog