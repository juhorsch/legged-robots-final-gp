import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jRotation
from ..constants import *


@jax.jit
def get_homo_transformation_matrix(R: jnp.array, t: jnp.array) -> jnp.array:
    """
    Calculates the homogeneous transformation matrix.

    Args:
        R (jnp.array): The rotation matrix.
        t (jnp.array): The translation vector.

    Returns:
        jnp.array: The homogeneous transformation matrix.

    """
    T1 = jnp.hstack([R, t[:, None]])
    T2 = jnp.array([0, 0, 0, 1])
    T = jnp.vstack([T1, T2])
    return T


@jax.jit
def homo_transformation(T: jnp.array, t: jnp.array) -> jnp.array:
    """
    Perform homogeneous transformation on a given point.

    Parameters:
    T (jnp.array): The transformation matrix.
    t (jnp.array): The point to be transformed.

    Returns:
    jnp.array: The transformed point.

    """
    th = T @ jnp.append(t, 1)
    return th[:3]


@jax.jit
def world_forward_kinematics(q: jnp.array) -> jnp.array:
    """
    Calculates the forward kinematics of a legged robot in the world frame.
    Args:
        q (jnp.array): Array of joint angles and body pose. The first 3 elements represent the body pose (x, y, z),
                       the next 3 elements represent the body orientation (roll, pitch, yaw), and the remaining
                       elements represent the joint angles of each leg.
    Returns:
        jnp.array: Array of 3D positions of each leg's foot in the world frame.
    """

    q_bp = q[0:3]
    q_rb = q[3:6]

    q_joints = jnp.reshape(q[6:], (NUM_LEGS, 3))

    X = [q_bp, q_rb]

    for i in range(NUM_LEGS):
        qi = q_joints[i]
        sign_di = params["sign_d"][i]
        sign_Li = params["sign_L"][i]

        o_R_b = jRotation.from_euler("xyz", q_rb).as_matrix()
        o_pos_b = q_bp
        o_T_b = get_homo_transformation_matrix(o_R_b, o_pos_b)

        b_R_h = jRotation.from_euler("x", qi[0]).as_matrix()
        b_pos_h = jnp.array([sign_Li * params["L"], sign_di * params["W"], 0])
        b_T_h = get_homo_transformation_matrix(b_R_h, b_pos_h)

        h_R_t = jRotation.from_euler("y", qi[1] - jnp.pi / 2).as_matrix()
        h_pos_t = jnp.array([0, params["d"] * sign_di, 0])
        h_T_t = get_homo_transformation_matrix(h_R_t, h_pos_t)

        t_R_c = jRotation.from_euler("y", qi[2]).as_matrix()
        t_pos_c = jnp.array(params["knee"])
        t_T_c = get_homo_transformation_matrix(t_R_c, t_pos_c)

        c_pos_f = jnp.array(params["foot"])
        o_pos_f = homo_transformation(o_T_b @ b_T_h @ h_T_t @ t_T_c, c_pos_f)
        X.append(o_pos_f)

    return jnp.hstack(X)


@jax.jit
def world_translation_jacobian(q: jnp.array) -> jnp.array:
    """
    Calculate the Jacobian matrix for the world translation of a legged robot.

    Parameters:
    - q (jnp.array): The joint angles of the robot.

    Returns:
    - jnp.array: The Jacobian matrix representing the derivative of the world translation with respect to the joint angles.
    """
    return jax.jacfwd(world_forward_kinematics)(q)


@jax.jit
def IKNewton(X: jnp.array, q0: jnp.array) -> jnp.array:
    """
    Performs inverse kinematics using the Newton method.
    Args:
        X (jnp.array): The desired end-effector position.
        q0 (jnp.array): The initial joint configuration.
    Returns:
        jnp.array: The joint configuration that achieves the desired end-effector position.
    """
    state = (q0, 1.0, 0)

    def body_fun(state):
        _q, _, _i = state
        _dq = jnp.linalg.solve(
            world_translation_jacobian(_q), (X - world_forward_kinematics(_q))
        )
        _q += _dq
        _step = jnp.linalg.norm(_dq)
        _i += 1
        return (_q, _step, _i)

    def cond_fun(state):
        _, _step, _i = state
        return (_i < params["IKN_MAX_IT"]) & (_step > params["IKN_EPS"])

    state = jax.lax.while_loop(cond_fun, body_fun, state)
    return state[0]
