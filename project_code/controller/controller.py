from scipy.spatial.transform import Rotation
import numpy as np
from ..state import State


def swing_leg_control_PD(
    state: State,
    o_pos_feet_d: np.array,
    o_vel_feet_d: np.array,
    o_acc_feet_d: np.array,
    stance_state: np.array,
) -> np.array:
    """
    Calculates the actuator torque for swing leg control using feedback linearisation.
    Parameters:
    state (State): The current state of the legged robot.
    o_pos_feet_d (np.array): The desired position of the feet.
    o_vel_feet_d (np.array): The desired velocity of the feet.
    o_acc_feet_d (np.array): The desired acceleration of the feet.
    stance_state (np.array): The stance state of the legs.
    Returns:
    np.array: The actuator torque for swing leg control.
    """

    K_d = np.diag(np.tile(np.array([300, 60, 250]), 4))
    K_p = np.diag(np.tile(np.array([10000, 8000, 10000]), 4))

    o_pos_feet_d = o_pos_feet_d.flatten()
    o_vel_feet_d = o_vel_feet_d.flatten()
    o_acc_feet_d = o_acc_feet_d.flatten()

    qvel = state.get_qvel()

    jac_t = state.get_jac_t_feet()
    o_pos_feet = state.get_feet_position().flatten()

    o_vel_feet = jac_t @ qvel

    M = state.get_M()
    h = state.get_force_bias()
    dJac_t = state.get_jac_t_dot_feet()

    o_vforce = (
        o_acc_feet_d
        + K_d @ (o_vel_feet_d - o_vel_feet)
        + K_p @ (o_pos_feet_d - o_pos_feet)
        - dJac_t @ qvel
    )
    qacc_d = np.linalg.pinv(jac_t) @ o_vforce
    tau_general = M @ qacc_d + h

    tau_actuator = state.get_actuator_torque(tau_general)

    swing_state = ~stance_state
    tau_actuator = tau_actuator * np.kron(swing_state, np.ones(3))

    return tau_actuator


def stance_leg_control(
    state: State, o_forces_d: np.array, stance_state: np.array
) -> np.array:
    """
    Calculates the actuator torque for the stance leg control.

    Parameters:
        state (State): The current state of the legged robot.
        o_forces_d (np.array): The desired forces in the leg frame.
        stance_state (np.array): The state of the stance leg.

    Returns:
        np.array: The actuator torque for the stance leg control.
    """
    
    o_forces_d = o_forces_d.flatten()
    jac_t = state.get_jac_t_feet()
    tau_general = -jac_t.T @ o_forces_d
    tau_actuator = state.get_actuator_torque(tau_general)
    tau_actuator = tau_actuator * np.kron(stance_state, np.ones(3))
    
    return tau_actuator


def leg_controller(
    state: State,
    o_pos_feet_d: np.array,
    o_vel_feet_d: np.array,
    o_acc_feet_d: np.array,
    o_forces_d: np.array,
    stance_state: np.array,
) -> np.array:
    """
    Calculates the total torque applied to the leg joints based on the desired foot positions, velocities, accelerations, forces, and the current stance state.
    Parameters:
    state (State): The current state of the leg.
    o_pos_feet_d (np.array): The desired positions of the feet.
    o_vel_feet_d (np.array): The desired velocities of the feet.
    o_acc_feet_d (np.array): The desired accelerations of the feet.
    o_forces_d (np.array): The desired forces applied to the feet.
    stance_state (np.array): The current stance state of the leg.
    Returns:
    np.array: The total torque applied to the leg joints.
    """
    tau_sw = swing_leg_control_PD(
        state, o_pos_feet_d, o_vel_feet_d, o_acc_feet_d, stance_state
    )
    tau_st = stance_leg_control(state, o_forces_d, stance_state)

    return tau_sw + tau_st