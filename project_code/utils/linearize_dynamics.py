import numpy as np
from scipy.spatial.transform import Rotation

from ..constants import *


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def rotate_inertia_and_invert(o_rot_b, In_b):
    """Compute the inertia matrix in the world frame."""
    o_R_b = Rotation.from_euler("xyz", o_rot_b, degrees=False).as_matrix()
    In_o = o_R_b @ In_b @ o_R_b.T
    In_o_inv = np.linalg.inv(In_o)
    return In_o, In_o_inv


def compute_r_cross_terms(t_mpc, In_o_inv, r_vectors):
    """Compute the r_cross terms using the skew matrix and Inertia inverse."""
    r_cross_terms = []
    for r in r_vectors:
        r_cross = t_mpc * np.dot(In_o_inv, skew(r))
        r_cross_terms.append(r_cross)
    return r_cross_terms


def build_A_matrix(R_z, t_mpc):
    """Construct the A matrix for the given time step."""
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    Z1 = np.zeros((3, 1))
    I1 = np.array([[0], [0], [1]])
    RZt = R_z.T * t_mpc

    A_matrix = np.block(
        [
            [I3, Z3, I3 * t_mpc, Z3, Z1],
            [Z3, I3, Z3, RZt, Z1],
            [Z3, Z3, I3, Z3, I1 * t_mpc],
            [Z3, Z3, Z3, I3, Z1],
            [Z1.T, Z1.T, Z1.T, Z1.T, 1],
        ]
    )
    return A_matrix


def build_B_matrix(r_cross_terms, t_mpc, m):
    """Construct the B matrix for the given time step."""
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    Z1 = np.zeros((1, 3))

    I3t_m = I3 * t_mpc / m

    B_matrix = np.block(
        [
            [Z3, Z3, Z3, Z3],
            [Z3, Z3, Z3, Z3],
            [I3t_m, I3t_m, I3t_m, I3t_m],
            r_cross_terms,
            [Z1, Z1, Z1, Z1],
        ]
    )
    return B_matrix


def get_AB_matrices(trajectory_state, trajectory_feet):
    """Generate Linear Dynamic System A and B matrices for the entire trajectory."""
    t_mpc = params["t_mpc"]
    m = params["mass"]
    In_b = np.array(params["In_b"])

    A_matrices = []
    B_matrices = []

    for i in range(len(trajectory_state)):
        cur_srb_state = trajectory_state[i]
        o_pos_feet = trajectory_feet[i]

        o_pos_b = cur_srb_state[0:3]
        o_rot_b = cur_srb_state[3:6]

        # Compute vector from COM to foot i
        r_vectors = [o_pos_feet[j] - o_pos_b for j in range(4)]

        # Compute inertia matrix and its inverse in world frame
        _, In_o_inv = rotate_inertia_and_invert(o_rot_b, In_b)

        # Compute r_cross terms
        r_cross_terms = compute_r_cross_terms(t_mpc, In_o_inv, r_vectors)

        # Compute rotation matrix for z-axis
        R_z = Rotation.from_euler("z", o_rot_b[2], degrees=False).as_matrix()

        # Build A and B matrices
        A_matrix = build_A_matrix(R_z, t_mpc)
        B_matrix = build_B_matrix(r_cross_terms, t_mpc, m)

        A_matrices.append(A_matrix)
        B_matrices.append(B_matrix)

    return np.stack(A_matrices), np.stack(B_matrices)


def build_constraint_matrices():
    """Build constant parts of the inequality constraint matrices."""

    mu = params["mu"]
    fmax = params["fmax"]

    Cil = np.array([[-1, 0, mu], [0, -1, mu], [1, 0, mu], [0, 1, mu], [0, 0, 1]])
    Ci = np.kron(np.eye(4), Cil)
    li = np.zeros((5 * 4, 1)).flatten()
    hi = np.array([np.inf, np.inf, np.inf, np.inf, fmax])
    hi = np.kron(np.ones(4), hi)
    return Ci, li, hi


def get_inequality_constraints(trajectory_stance_state):
    """Generate inequality constraint matrices A and b for the entire trajectory."""
    n = len(trajectory_stance_state)
    dc = 5 * NUM_LEGS
    du = 3 * NUM_LEGS
    C = np.zeros((n * dc, du * n))
    l = []
    h = []

    Ci, li, hi_pattern = build_constraint_matrices()

    for i in range(n):
        in_stance = np.kron((trajectory_stance_state[i]), np.ones(5))

        # If leg is in swing, upper bound is 0!
        hi = hi_pattern * 1
        hi[in_stance == 0] = 0

        C[i * dc : (i + 1) * dc, i * du : (i + 1) * du] = Ci
        l.append(li)
        h.append(hi)

    l = np.hstack(l)
    h = np.hstack(h)

    return C, l, h
