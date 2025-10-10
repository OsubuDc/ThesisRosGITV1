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
    
    def check_reachability(self):
        """Check if target is within workspace"""
        # Test multiple configurations to find workspace bounds
        max_reach = 0
        min_reach = float('inf')
        
        for q1 in np.linspace(self.model.lowerPositionLimit[0], self.model.upperPositionLimit[0], 10):
            for q2 in np.linspace(self.model.lowerPositionLimit[1], self.model.upperPositionLimit[1], 10):
                q_test = np.array([q1, q2])
                pin.forwardKinematics(self.model, self.data, q_test)
                pin.updateFramePlacements(self.model, self.data)
                x_test = self.data.oMf[self.ee_id].translation[:2]
                reach = np.linalg.norm(x_test)
                max_reach = max(max_reach, reach)
                min_reach = min(min_reach, reach)
        
        target_dist = np.linalg.norm(self.x_des)
        self.get_logger().info(f'Workspace: min_reach={min_reach:.3f}m, max_reach={max_reach:.3f}m')
        self.get_logger().info(f'Target distance: {target_dist:.3f}m')
        
        if target_dist > max_reach:
            self.get_logger().warn(f'⚠️  TARGET UNREACHABLE! Target is {target_dist:.3f}m but max reach is {max_reach:.3f}m')
        elif target_dist < min_reach:
            self.get_logger().warn(f'⚠️  TARGET TOO CLOSE! Target is {target_dist:.3f}m but min reach is {min_reach:.3f}m')
        else:
            self.get_logger().info('✓ Target is within workspace')
    
    def robot_dynamics(self, t, y):
        """Compute derivatives for the integrator"""
        q = y[:self.model.nq]
        v = y[self.model.nq:]
        
        # Clamp joint positions to limits
        q_clamped = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        
        # If at limits, zero out velocity in that direction
        v_clamped = v.copy()
        for i in range(self.model.nq):
            if q_clamped[i] <= self.model.lowerPositionLimit[i] and v[i] < 0:
                v_clamped[i] = 0  # Stop moving into lower limit
            elif q_clamped[i] >= self.model.upperPositionLimit[i] and v[i] > 0:
                v_clamped[i] = 0  # Stop moving into upper limit
        
        # Forward kinematics with clamped values
        pin.forwardKinematics(self.model, self.data, q_clamped, v_clamped)
        pin.updateFramePlacements(self.model, self.data)
        
        # Jacobian and its derivative (use clamped q)
        J = pin.computeFrameJacobian(self.model, self.data, q_clamped, self.ee_id)[:2, :]
        J_dot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.ee_id, pin.WORLD)[:2, :]
        
        # End effector position and errors
        x = self.data.oMf[self.ee_id].translation[:2]
        x_err = self.x_des - x
        xdot_err = self.xdot_des - J @ v_clamped
        
        # Desired EE acceleration (PD control in task space)
        x_acc_des = self.xddot_des + self.Kd @ xdot_err + self.Kp @ x_err
        
        # Inverse dynamics (use clamped values)
        B = pin.crba(self.model, self.data, q_clamped)  # Mass matrix
        n = pin.rnea(self.model, self.data, q_clamped, v_clamped, np.zeros(self.model.nv))  # Nonlinear terms
        
        # Map task-space acceleration to joint-space
        qddot_task = np.linalg.pinv(J, rcond=1e-2) @ (x_acc_des - J_dot @ v_clamped)
        u = B @ qddot_task + n  # Control torques
        
        # Actual joint accelerations from dynamics
        qddot = np.linalg.solve(B, u - n)
        
        # Zero out acceleration if hitting limits
        for i in range(self.model.nq):
            if q_clamped[i] <= self.model.lowerPositionLimit[i] and qddot[i] < 0:
                qddot[i] = 0
            elif q_clamped[i] >= self.model.upperPositionLimit[i] and qddot[i] > 0:
                qddot[i] = 0
        
        return np.concatenate((v_clamped, qddot))
    
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
            
            # Check if joints are at limits
            at_limit = []
            for i in range(self.model.nq):
                if abs(q[i] - self.model.lowerPositionLimit[i]) < 0.01:
                    at_limit.append(f'q{i+1}=LOWER')
                elif abs(q[i] - self.model.upperPositionLimit[i]) < 0.01:
                    at_limit.append(f'q{i+1}=UPPER')
            limit_str = f' [{", ".join(at_limit)}]' if at_limit else ''
            
            self.get_logger().info(
                f't={self.time:.2f}s | q=[{q[0]:+.2f}, {q[1]:+.2f}] | '
                f'EE XY: [{x[0]:+.3f}, {x[1]:+.3f}] | '
                f'Err: [{x_err[0]:+.4f}, {x_err[1]:+.4f}]{limit_str}'
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
    
# python3 pendulum_sim_node.py
#   ↓
# main() called
#   ↓
# rclpy.init()  ← ROS 2 starts, connects to daemon
#   ↓
# node = PendulumSimulator()  ← __init__() runs
#   ├─ Parameters declared
#   ├─ URDF loaded
#   ├─ Publisher created
#   └─ Timer started (calls step_and_publish every 0.01s)
#   ↓
# rclpy.spin(node)  ← **BLOCKS HERE** - event loop runs
#   │
#   │ (Timer fires every 0.01s in background)
#   │   └─> step_and_publish() → integrate → publish
#   │   └─> step_and_publish() → integrate → publish
#   │   └─> step_and_publish() → integrate → publish
#   │   ...
#   │
#   ↓ (User presses Ctrl+C)
# except KeyboardInterrupt
#   ↓
# finally:
#   ├─ node.destroy_node()  ← Cleanup resources
#   └─ rclpy.shutdown()     ← Disconnect from ROS 2

# ┌─────────────────────────────────────────────────────┐
# │           ROS 2 Node: PendulumSimulator             │
# ├─────────────────────────────────────────────────────┤
# │  __init__():                                        │
# │    1. Declare parameters (urdf_path, mesh_dir)     │
# │    2. Load Pinocchio model                         │
# │    3. Create publisher (joint_states topic)        │
# │    4. Start timer (100 Hz → step_and_publish)      │
# ├─────────────────────────────────────────────────────┤
# │  step_and_publish() [called every 0.01s]:          │
# │    1. Integrate dynamics (solve_ivp)               │
# │    2. Build JointState message                     │
# │    3. Publish to /joint_states topic               │
# ├─────────────────────────────────────────────────────┤
# │  robot_dynamics():                                  │
# │    - Your control law (same as before)             │
# │    - Returns [v, qddot] for integrator             │
# └─────────────────────────────────────────────────────┘
#          ↓ publishes to
#     /joint_states topic
#          ↓
#     robot_state_publisher (subscribes)
#          ↓
#     Computes TF transforms
#          ↓
#     RViz2 (subscribes to TF)
#          ↓
#     Visualizes robot!