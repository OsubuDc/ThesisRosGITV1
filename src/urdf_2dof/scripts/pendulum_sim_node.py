#!/usr/bin/env python3

import rclpy # ROS2 functinoality: Nodes, publishers, timers
from rclpy.node import Node # Base class for nodes
from sensor_msgs.msg import JointState # Message type for joint data
from std_msgs.msg import Header #Message header: timestamp, frame_id
import numpy as np # Numerical computation
import pinocchio as pin #Rigid body dynamics
from pinocchio.robot_wrapper import RobotWrapper
from scipy.integrate import solve_ivp #ODE solver


# Class inherits from Node, gets all functions, variables, behaviours that are defined in the Node class
# Class is a blueprint for creating objects. An object is an instance of a class.
class PendulumSimulator(Node):
    # When new object is created from class, it gets passed down to the first argument 'self'.
    def __init__(self):
        #Calls parent class constructor, registers this node with ROS 2 as 'pendulum_simulator'
        # Run _init_ of the parent class 'Node' and 'pendulum_simulator' is passed as the argument
        # Node.__init__('the name of the node that needs to be created)
            # ros2 node list -> to see al the nodes
            # rqt_graph -> see network of nodes and topics
        # Each node needs a unique name
        super().__init__('pendulum_simulator')
        
        # Declare parameters

        # Two-Step Process in ROS 2:
        # Step 1: declare_parameter(name, default_value)

            # Registers the parameter with ROS 2
            # Sets a default value if not provided externally
            # Makes the parameter discoverable (you can see it with ros2 param list)

        # Step 2: get_parameter(name).value

            # Retrieves the actual value
            # .value extracts the data (it's wrapped in a Parameter object)

        # ros2 param list
        # ros2 param get /pendulum_simulator urdf_path

        # parameter declared in the node
        self.declare_parameter('urdf_path', '/ThesisRosGITV1/src/urdf_2dof/urdf/2dof.urdf')
        self.declare_parameter('mesh_dir', '/ThesisRosGITV1/src/urdf_2dof/meshes')
        
        # Store parameter value in python variable
        urdf_path = self.get_parameter('urdf_path').value
        mesh_dir = self.get_parameter('mesh_dir').value
        
        self.get_logger().info(f'Loading URDF from: {urdf_path}')
        
        # Load robot model
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
        self.model, self.data = self.robot.model, self.robot.data
        self.model.gravity.linear = np.array([0.0, -9.81, 0.0])
        
        # Get joint names
        self.joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]
        self.get_logger().info(f'Joint names: {self.joint_names}')
        
        # End effector frame
        self.ee_frame = "EndEffector"
        try:
            self.ee_id = self.model.getFrameId(self.ee_frame)
            self.get_logger().info(f'Found end effector: {self.ee_frame}')
        except:
            self.get_logger().error(f'Could not find frame "{self.ee_frame}"')
            self.get_logger().info(f'Available: {[self.model.frames[i].name for i in range(self.model.nframes)]}')
            self.ee_id = self.model.nframes - 1
            self.get_logger().warn(f'Using: {self.model.frames[self.ee_id].name}')
        
        # target 
        self.x_des = np.array([0.2, 0.05])
        self.xdot_des = np.zeros(2)
        self.xddot_des = np.zeros(2)
        
        # PD gains
        self.Kp = np.diag([50, 50])
        self.Kd = np.diag([40, 40])
                
        # ENFORCE JOINT LIMITS (critical to stop spinning!)
        self.q_min = np.array([-np.pi, -np.pi])
        self.q_max = np.array([np.pi, np.pi])
        
        self.get_logger().info(f'Target: [{self.x_des[0]:.3f}, {self.x_des[1]:.3f}]')
        self.get_logger().warn(f'Joint limits: [{np.degrees(self.q_min[0]):.0f}°, {np.degrees(self.q_min[1]):.0f}°] to [{np.degrees(self.q_max[0]):.0f}°, {np.degrees(self.q_max[1]):.0f}°]')
        
        # Simulation parameters
        self.dt = 0.01
        self.time = 0.0
        
        # Initial state
        q_init = np.array([0.01, 0.01]) 
        self.y = np.concatenate([q_init, np.zeros(2)])  # Zero initial velocity
        
        self.get_logger().info(f'Initial config: [{np.degrees(q_init[0]):.1f}°, {np.degrees(q_init[1]):.1f}°]')
        
        # Publisher
        # create attribute of the object; joint_pub
        # Publisher sends messages via topic. Type of message is JointState (standard for robot joint positions/velocities)
        # joint_states is the name of the topic the node will publish to. Other nodes can subscripe to this topic (rviz, robot_state_publisher (converts joint states to TF transforms))
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer
        # Timers fires every dt seconds, calls step_and_publish method (function that belongs to class)
        self.timer = self.create_timer(self.dt, self.step_and_publish)
        
        self.get_logger().info('Pendulum simulator started!')
    
    def robot_dynamics(self, t, y):
        q = y[:self.model.nq]
        v = y[self.model.nq:]
        
        # Forward kinematics
        # Computes link positions for current q. V needed as input for J_dot
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        
        # Jacobian
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id)[:2, :]
        J_dot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.ee_id, pin.WORLD)[:2, :]
        
        # EE position and errors
        # oMf = origin to frame transformation matrices for all frames
        # translation -> ony x y z part of the matrix
        # :2 -> x y
        x = self.data.oMf[self.ee_id].translation[:2]
        x_err = self.x_des - x
        xdot_err = self.xdot_des - J @ v
        
        # Desired EE acceleration
        x_acc_des = self.xddot_des + self.Kd @ xdot_err + self.Kp @ x_err
        
        # Inverse dynamics
        #crba : Composite Rigid Body Algorithm
        #rnea : Recursive Newton-Euler Algorithm
        B = pin.crba(self.model, self.data, q)
        n = pin.rnea(self.model, self.data, q, v, np.zeros(self.model.nv))
        
        # Map task-space to joint-space
        # x¨=J(q)q¨​+J˙(q,q˙​)q˙ -> q¨​task​=J+(x¨des​−Jdot​v)
        #pinv = pseudo inv, inv = inv (compare)
        #qddot_task = np.linalg.pinv(J, rcond=1e-2) @ (x_acc_des - J_dot @ v)
        qddot_task = np.linalg.inv(J) @ (x_acc_des - J_dot @ v)
        # Control torque 
        u = B @ qddot_task + n
        
        # Actual joint accelerations
        qddot = np.linalg.solve(B, u - n)
        
        return np.concatenate((v, qddot))
    

    
    def step_and_publish(self):

        # CLAMP state BEFORE integration -> start within limits
        self.y[:2] = np.clip(self.y[:2], self.q_min, self.q_max)
        
        # Integrate
        sol = solve_ivp(
            self.robot_dynamics,
            [self.time, self.time + self.dt],
            self.y,
            method='RK45',
            max_step=0.001  # Same as your working code
        )
        
        # Update state
        self.y = sol.y[:, -1] #final value of the integration
        
        # CLAMP state AFTER integration (CRITICAL!) -> end within limits
        self.y[:2] = np.clip(self.y[:2], self.q_min, self.q_max)
        
        # Zero velocity if at limits
        for i in range(2):
            if self.y[i] <= self.q_min[i] and self.y[2+i] < 0:
                self.y[2+i] = 0  # Stop moving into lower limit
            elif self.y[i] >= self.q_max[i] and self.y[2+i] > 0:
                self.y[2+i] = 0  # Stop moving into upper limit
        
        self.time += self.dt
        
        # Extract state
        q = self.y[:self.model.nq]
        v = self.y[self.model.nq:]
        
        # Publish
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()
        msg.velocity = v.tolist()
        msg.effort = []
        
        self.joint_pub.publish(msg)
        
        # Log every second WITH JOINT ANGLES
        if int(self.time * 100) % 100 == 0:
            pin.forwardKinematics(self.model, self.data, q, v)
            pin.updateFramePlacements(self.model, self.data)
            x = self.data.oMf[self.ee_id].translation[:2]
            x_err = self.x_des - x
            
            # Convert to degrees
            q_deg = np.degrees(q)
            
            # Check if at limits
            limits = []
            if abs(q[0] - self.q_min[0]) < 0.02:
                limits.append('J1=MIN')
            elif abs(q[0] - self.q_max[0]) < 0.02:
                limits.append('J1=MAX')
            if abs(q[1] - self.q_min[1]) < 0.02:
                limits.append('J2=MIN')
            elif abs(q[1] - self.q_max[1]) < 0.02:
                limits.append('J2=MAX')
            
            limit_str = f' [{", ".join(limits)}]' if limits else ''
            
            self.get_logger().info(
                f't={self.time:5.2f}s | '
                f'q=[{q_deg[0]:+6.1f}°, {q_deg[1]:+6.1f}°] | '
                f'EE=[{x[0]:+.3f}, {x[1]:+.3f}] | '
                f'err={np.linalg.norm(x_err):.4f}{limit_str}'
            )

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PendulumSimulator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.destroy_node()
                rclpy.shutdown()
        except:
            pass

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