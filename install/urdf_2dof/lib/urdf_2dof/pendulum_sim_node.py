#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.integrate import solve_ivp

class PendulumSimulator(Node):
    def __init__(self):
        super().__init__('pendulum_simulator')
        
        # Declare and get parameters
        self.declare_parameter('urdf_path', '/ThesisRosGITV1/src/urdf_2dof/urdf/2dof.urdf')
        self.declare_parameter('mesh_dir', '/ThesisRosGITV1/src/urdf_2dof/meshes')
        
        urdf_path = self.get_parameter('urdf_path').value
        mesh_dir = self.get_parameter('mesh_dir').value
        
        self.get_logger().info(f'Loading URDF from: {urdf_path}')
        
        # Load robot model
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
        self.model, self.data = self.robot.model, self.robot.data
        self.model.gravity.linear = np.array([0.0, -9.81, 0.0])
        
        # Get joint names from model (skip universe at index 0)
        self.joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]
        self.get_logger().info(f'Joint names: {self.joint_names}')
        
        # End effector frame
        self.ee_frame = "EndEffector"
        try:
            self.ee_id = self.model.getFrameId(self.ee_frame)
        except:
            self.get_logger().error(f'Could not find frame "{self.ee_frame}" in URDF!')
            self.get_logger().info(f'Available frames: {[self.model.frames[i].name for i in range(self.model.nframes)]}')
            # Fallback: use last frame
            self.ee_id = self.model.nframes - 1
            self.get_logger().warn(f'Using frame: {self.model.frames[self.ee_id].name}')
        
        # Desired target in XY plane
        self.x_des = np.array([0.05, 0.05])
        self.xdot_des = np.zeros(2)
        self.xddot_des = np.zeros(2)
        
        # PD gains
        self.Kp = np.diag([100, 100])
        self.Kd = np.diag([20, 20])
        
        # Simulation parameters
        self.dt = 0.01  # 100 Hz
        self.time = 0.0
        
        # Initial state: [q1, q2, v1, v2]
        self.y = np.array([0.2, 0.1, 0.0, 0.0])
        
        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer for simulation loop
        self.timer = self.create_timer(self.dt, self.step_and_publish)
        
        self.get_logger().info('Pendulum simulator started!')
    
    def robot_dynamics(self, t, y):
        """Compute derivatives for the integrator"""
        q = y[:self.model.nq]
        v = y[self.model.nq:]
        
        # Forward kinematics
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        
        # Jacobian and its derivative
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id)[:2, :]
        J_dot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.ee_id, pin.WORLD)[:2, :]
        
        # End effector position and errors
        x = self.data.oMf[self.ee_id].translation[:2]
        x_err = self.x_des - x
        xdot_err = self.xdot_des - J @ v
        
        # Desired EE acceleration (PD control in task space)
        x_acc_des = self.xddot_des + self.Kd @ xdot_err + self.Kp @ x_err
        
        # Inverse dynamics
        B = pin.crba(self.model, self.data, q)  # Mass matrix
        n = pin.rnea(self.model, self.data, q, v, np.zeros(self.model.nv))  # Nonlinear terms
        
        # Map task-space acceleration to joint-space
        qddot_task = np.linalg.pinv(J, rcond=1e-2) @ (x_acc_des - J_dot @ v)
        u = B @ qddot_task + n  # Control torques
        
        # Actual joint accelerations from dynamics
        qddot = np.linalg.solve(B, u - n)
        
        return np.concatenate((v, qddot))
    
    def step_and_publish(self):
        """Integrate one timestep and publish joint state"""
        # Integrate from current time to time + dt
        sol = solve_ivp(
            self.robot_dynamics,
            [self.time, self.time + self.dt],
            self.y,
            method='RK45',
            max_step=self.dt / 10
        )
        
        # Update state
        self.y = sol.y[:, -1]
        self.time += self.dt
        
        # Extract q and v
        q = self.y[:self.model.nq]
        v = self.y[self.model.nq:]
        
        # Publish joint state
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()
        msg.velocity = v.tolist()
        msg.effort = []
        
        self.joint_pub.publish(msg)
        
        # Log EE position periodically (every 1 second)
        if int(self.time * 100) % 100 == 0:  # Every 100 steps = 1 second
            pin.forwardKinematics(self.model, self.data, q, v)
            pin.updateFramePlacements(self.model, self.data)
            x = self.data.oMf[self.ee_id].translation[:2]
            x_err = self.x_des - x
            self.get_logger().info(
                f't={self.time:.2f}s | EE XY: [{x[0]:+.3f}, {x[1]:+.3f}] | '
                f'Err: [{x_err[0]:+.4f}, {x_err[1]:+.4f}]'
            )

def main(args=None):
    rclpy.init(args=args)
    node = PendulumSimulator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
