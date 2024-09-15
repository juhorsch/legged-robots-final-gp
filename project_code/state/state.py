from scipy.spatial.transform import Rotation
import numpy as np
from ..constants import *
import mujoco

class State:
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def get_body_position(self):
        return self.data.qpos[0:3] * 1
    
    def get_body_velocity(self):
        return self.data.qvel[0:3] * 1
    
    def get_body_angular_velocity(self):
        return self.data.qvel[3:6] * 1
    
    def get_body_rotation_euler(self):
        quat = self.data.qpos[3:7] * 1
        euler = Rotation.from_quat(quat, scalar_first=True).as_euler('xyz')
        return euler
    
    def get_feet_position(self):
        feet_pos = np.vstack([self.data.geom('FL').xpos * 1, 
                              self.data.geom('FR').xpos * 1, 
                              self.data.geom('RL').xpos * 1,
                              self.data.geom('RR').xpos * 1])
        return feet_pos
    
    def time(self):
        return self.data.time * 1
    
    def update_time(self, dt):
        self.data.time += dt
    
    def get_qpos_euler(self):
        euler = self.get_body_rotation_euler()
        qpos = np.hstack([self.data.qpos[:3], euler, self.data.qpos[7:]])
        return qpos
    
    def set_qpos_euler(self, qpos):
        euler = qpos[3:6] * 1
        quat = Rotation.from_euler('xyz', euler).as_quat(scalar_first=True)
        self.data.qpos = np.hstack([qpos[:3] * 1, quat, qpos[6:] * 1])
    
    def get_data(self):
        return self.data
    
    def update_state(self, new_data):
        self.data = new_data
        
    def get_qpos_joints(self):
        return self.data.qpos[7:] * 1
    
    def get_qvel_joints(self):
        return self.data.qvel[6:] * 1
    
    def get_qpos(self):
        return self.data.qpos * 1
    
    def get_qvel(self):
        return self.data.qvel * 1
    
    def get_xpos(self):
        base_pos = self.get_body_position()
        base_rot = self.get_body_rotation_euler()
        feet_pos = self.get_feet_position().flatten()
        return np.hstack([base_pos, base_rot, feet_pos]) 
    
    
    def get_xvel(self):
        base_vel = self.get_body_velocity()
        base_omega = self.get_body_angular_velocity()
        feet_vel = self.get_jac_t_feet() @ self.get_qvel()
        return np.hstack([base_vel, base_omega, feet_vel]) 
           
    def get_body_state(self):
        return np.hstack([self.get_body_position(),
                          self.get_body_rotation_euler(),
                          self.get_body_velocity(),
                          self.get_body_angular_velocity()])
        
    def set_body_vel(self, vel):
        self.data.qvel[0:3] = vel
        
    def set_body_angular_vel(self, vel):
        self.data.qvel[3:6] = vel
        
    def set_body_rot(self, euler):
        quat = Rotation.from_euler('xyz', euler).as_quat(scalar_first=True)
        self.data.qpos[3:7] = quat
        
    def get_jac_t_feet(self):
        jac_t = []
        nv = self.model.nv
        ids = [self.data.geom('FL').id,
               self.data.geom('FR').id,
               self.data.geom('RL').id,
               self.data.geom('RR').id]
        
        for id in ids:
            jac_ti = np.zeros((3, nv))
            mujoco.mj_jacGeom(self.model, self.data, jac_ti, None, id)
            jac_t.append(jac_ti)
        
        jac_t = np.vstack(jac_t) * 1
        return jac_t
    
    def get_jac_t_dot_feet(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        qpos0 = self.data.qpos * 1
        qvel0 = self.data.qvel * 1     
        jac_t_0 = self.get_jac_t_feet()
        h = 1e-10
        mujoco.mj_integratePos(self.model, self.data.qpos, self.data.qvel, h)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jact_t_1 = self.get_jac_t_feet()
        jac_dt = (jact_t_1 - jac_t_0) / h
        self.data.qpos = qpos0
        self.data.qvel = qvel0
        return jac_dt
    
    def get_force_bias(self):
        return self.data.qfrc_bias * 1
    
    def get_M(self):
        full_inertia_matrix = np.zeros((self.model.nv, self.model.nv))

        # Convert the sparse inertia matrix to a full matrix
        mujoco.mj_fullM(self.model, full_inertia_matrix, self.data.qM)

        return full_inertia_matrix
            
    def get_actuator_torque(self, tau_general):
        return (np.atleast_2d(tau_general) @ np.linalg.pinv(self.data.actuator_moment)).flatten() * 1
    
        
        
    
        