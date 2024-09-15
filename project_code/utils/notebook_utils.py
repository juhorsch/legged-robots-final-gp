import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(trajectory_position, trajectory_velocity, time_steps):
    pos_x = trajectory_position[:, 0, 0]
    pos_y = trajectory_position[:, 0, 1]
    rot_yaw = trajectory_position[:, 1, 2]
    
    vel_x = trajectory_velocity[:, 0, 0]
    vel_y = trajectory_velocity[:, 0, 1]
    omg_yaw = trajectory_velocity[:, 1, 2]
    
    vel = np.sqrt(vel_x**2 + vel_y**2)
    
    # Plot X over Y, Next to X over Time (Red) and Y over Time (Blue). Underneath Plot Yaw over Time and Next to it plot Velocity over Time and angular velocity over time.
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(pos_x, pos_y)
    axs[0, 0].set_title('X over Y')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    
    axs[0, 1].plot(time_steps, pos_x, 'r')
    axs[0, 1].plot(time_steps, pos_y, 'b')
    axs[0, 1].set_title('X over Time (Red) and Y over Time (Blue)')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('X, Y')
    
    axs[1, 0].plot(time_steps, rot_yaw)
    axs[1, 0].set_title('Yaw over Time')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Yaw')
    
    axs[1, 1].plot(time_steps, vel, 'r')
    axs[1, 1].plot(time_steps, omg_yaw, 'b')
    axs[1, 1].set_title('Velocity over Time (Red) and Angular Velocity over Time (Blue)')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Velocity')
    
    plt.show()
    
def plot_sw_prog(sw_prog, stance_state, time_steps):
    fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
    fig.suptitle("Gait Planner Visualization")
    
    # Calculate the min and max of swing progress for setting dynamic y-limits
    sw_prog_min = sw_prog.min()
    sw_prog_max = sw_prog.max()

    # Plot Stance State on the left (first column)
    for i in range(4):
        axes[i, 0].step(time_steps, stance_state[:, i], where='mid', label=f"Leg {i+1} Stance State")
        axes[i, 0].set_ylabel(f'Leg {i+1}')
        axes[i, 0].set_ylim(-0.1, 1.1)
        axes[i, 0].set_yticks([0, 1])
        axes[i, 0].legend(loc='upper right')

    # Plot Swing Progress on the right (second column) and add a horizontal line at y = 0
    for i in range(4):
        axes[i, 1].plot(time_steps, sw_prog[:, i], label=f"Leg {i+1} Swing Progress")
        axes[i, 1].axhline(0, color='gray', linestyle='--', linewidth=1)  # Horizontal line at y=0
        axes[i, 1].set_ylim(sw_prog_min - 0.2, sw_prog_max + 0.2)
        axes[i, 1].legend(loc='upper right')

    # Set labels
    for ax in axes[:, 0]:
        ax.set_ylabel('State')
    
    for ax in axes[:, 1]:
        ax.set_ylabel('Swing Progress')

    # Only the bottom row gets the time label
    for ax in axes[-1, :]:
        ax.set_xlabel('Time Steps')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_t_ab(t_ab, period, time_steps):
    t_st = t_ab[:, 1] - t_ab[:, 0]
    t_sw = period - t_st
    
    t_dash = np.clip(np.mod(time_steps[:, None] - t_ab[:, 0][None, :], period), 0, 1)
    sw_prog = 1 / (t_sw[None, :]) * (t_dash - t_st[None, :])
    
    sw_prog = (sw_prog >= 0)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Gait Planner Visualization")
    
    # Calculate the min and max of swing progress for setting dynamic y-limits
    sw_prog_min = sw_prog.min()
    sw_prog_max = sw_prog.max()

    # Plot Swing Progress on the right (second column) and add a horizontal line at y = 0
    for i in range(4):
        axes[i].plot(time_steps, sw_prog[:, i], label=f"Leg {i+1} Swing State")
        axes[i].fill_between(time_steps, sw_prog[:, i], 0, color='blue', alpha=.1)
        axes[i].set_ylim(sw_prog_min - 0.2, sw_prog_max + 0.2)
        axes[i].legend(loc='upper right')
    

    # Only the bottom row gets the time label
    for ax in axes[:]:
        ax.set_xlabel('Time Steps')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_t(t_dash, t_ab, period, time_steps):
    t_st = t_ab[:, 1] - t_ab[:, 0]
    t_sw = period - t_st
    
    t = np.clip(np.mod(time_steps[:, None] - t_ab[:, 0][None, :], period), 0, 1)
    sw_prog = 1 / (t_sw[None, :]) * (t- t_st[None, :])
    
    sw_prog = (sw_prog >= 0)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Gait Planner Visualization")
    
    # Calculate the min and max of swing progress for setting dynamic y-limits
    sw_prog_min = t_dash.min()
    sw_prog_max = t_dash.max()

    # Plot Swing Progress on the right (second column) and add a horizontal line at y = 0
    for i in range(4):
        axes[i].plot(time_steps, sw_prog[:, i], label=f"Leg {i+1} Swing State")
        axes[i].plot(time_steps, t_dash[:, i], label=f"Leg {i+1} t")
        axes[i].fill_between(time_steps, sw_prog[:, i], 0, color='blue', alpha=.1)
        axes[i].set_ylim(sw_prog_min - 0.2, sw_prog_max + 0.2)
        axes[i].legend(loc='upper right')

    # Only the bottom row gets the time label
    for ax in axes[:]:
        ax.set_xlabel('Time Steps')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_polynomials():
    coef_x = np.array([-2, 3, 0, 0])
    coef_z1 = np.array([-16 * 1, 12 * 1, 0, 0])
    coef_z2 = np.array([-16 * -1, 36 * -1, -24 * -1, 1 + 5 * -1])
    
    coef_vx = np.array([0, -6, 6, 0])
    coef_vz1 = np.array([0, -16 * 3, 24 * 1, 0])
    coef_vz2 = np.array([0, -16 * -3, 36 * -2, -24 * -1])
    
    t = np.linspace(0, 1, 1000)
    
    def eval_poly(coef, t):
        return coef[3] + coef[2] * t + coef[1] * t**2 + coef[0] * t**3
    
    x = eval_poly(coef_x, t)
    z1 = eval_poly(coef_z1, t[:500])
    z2 = eval_poly(coef_z2, t[500:])
    z = np.concatenate([z1, z2])
    
    vx = eval_poly(coef_vx, t)
    vz1 = eval_poly(coef_vz1, t[:500])
    vz2 = eval_poly(coef_vz2, t[500:])
    vz = np.concatenate([vz1, vz2])
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(t, x)
    axs[0, 0].set_title('X and Y')
    axs[0, 0].set_xlabel('Time')
    
    axs[0, 1].plot(t, z)
    axs[0, 1].set_title('Z')
    axs[0, 1].set_xlabel('Time')
    
    axs[1, 0].plot(t, vx)
    axs[1, 0].set_title('Velocity X and Y')
    axs[1, 0].set_xlabel('Time')
    
    axs[1, 1].plot(t, vz)
    axs[1, 1].set_title('Velocity Z')
    axs[1, 1].set_xlabel('Time')
    
    plt.show()
    
def plot_desired_vs_measured(pos_ds, pos_ms, vel_ds, vel_ms):
    
    pos_ds = np.stack(pos_ds)
    pos_ms = np.stack(pos_ms)
    vel_ds = np.stack(vel_ds)
    vel_ms = np.stack(vel_ms)


    fig, axes = plt.subplots(6, 3, figsize=(15, 18))
    fig.suptitle('Desired vs Measured Path (Position and Velocity)', fontsize=16)

    labels = [
        'Base Position X', 'Base Position Y', 'Base Position Z',
        'Base Orientation Roll', 'Base Orientation Pitch', 'Base Orientation Yaw',
        'FL Position X', 'FL Position Y', 'FL Position Z',
        'FR Position X', 'FR Position Y', 'FR Position Z',
        'RL Position X', 'RL Position Y', 'RL Position Z',
        'RR Position X', 'RR Position Y', 'RR Position Z'
    ]

    # Plot positions (base pos, orientation, legs)
    for i in range(18):
        row, col = divmod(i, 3)
        axes[row, col].plot(pos_ds[:, i], label='Desired Position')
        axes[row, col].plot(pos_ms[:, i], label='Measured Position')
        axes[row, col].set_title(labels[i])
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel('Position')
        axes[row, col].legend()

    # Add a figure for velocities
    fig_vel, axes_vel = plt.subplots(6, 3, figsize=(15, 18))
    fig_vel.suptitle('Desired vs Measured Path (Velocities)', fontsize=16)

    vel_labels = [
        'Base Velocity X', 'Base Velocity Y', 'Base Velocity Z',
        'Angular Velocity Roll', 'Angular Velocity Pitch', 'Angular Velocity Yaw',
        'FL Velocity X', 'FL Velocity Y', 'FL Velocity Z',
        'FR Velocity X', 'FR Velocity Y', 'FR Velocity Z',
        'RL Velocity X', 'RL Velocity Y', 'RL Velocity Z',
        'RR Velocity X', 'RR Velocity Y', 'RR Velocity Z'
    ]

    # Plot velocities (base vel, angular vel, legs)
    for i in range(18):
        row, col = divmod(i, 3)
        axes_vel[row, col].plot(vel_ds[:, i], label='Desired Velocity')
        axes_vel[row, col].plot(vel_ms[:, i], label='Measured Velocity')
        axes_vel[row, col].set_title(vel_labels[i])
        axes_vel[row, col].set_xlabel('Time Step')
        axes_vel[row, col].set_ylabel('Velocity')
        axes_vel[row, col].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_mpc(title, xk, X, Xd, U, l = None, h = None):
    # check if dimensions are more than 2
    nx = X.shape[1]
    nu = U.shape[1]
    if nx > 3 or nu > 3:
        print("Too many dimensions to plot")
        return
    
    if nu != nx:
        print("Number of inputs and states must be the same")
        return
    
    # 1 subplot for each state and input
    fig, axs = plt.subplots(nx, 2, figsize=(10, 8))
    fig.suptitle(title)

    # add xk and remove last element
    X = X[:-1, :]
    X = np.vstack((xk, X))
    U = U[:-1, :]

    # Plot states and inputs
    for i in range(nx):
        axs[i, 0].plot(Xd[:, i], label=f'Desired State {i+1}')
        axs[i, 0].plot(X[:, i], label=f'Measured State {i+1}')
        axs[i, 0].set_title(f'State {i+1}')
        axs[i, 0].set_xlabel('Time Step')
        axs[i, 0].set_ylabel('State Value ')
        axs[i, 0].legend()

        axs[i, 1].plot(U[:, i], label=f'Input {i+1}')
        axs[i, 1].set_title(f'Input {i+1}')
        axs[i, 1].set_xlabel('Time Step')
        axs[i, 1].set_ylabel('Input Value')
        
        # if constraints are provided add horizontal lines
        if l is not None:
            axs[i, 1].axhline(l[i], color='b', linestyle='--', label='Lower Bound')
        if h is not None:
            axs[i, 1].axhline(h[i], color='r', linestyle='--', label='Upper Bound')
        
        axs[i, 1].legend()

    plt.tight_layout()
    plt.show()


        
    
    