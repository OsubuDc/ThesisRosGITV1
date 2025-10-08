import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.integrate import solve_ivp

# --- Load robot model ---
urdf_path = "/ThesisRosGITV1/V2_URDF/urdf/V2_URDF_fixed.urdf"
mesh_dir = "/ThesisRosGITV1/V2_URDF/meshes"
robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
model, data = robot.model, robot.data
model.gravity.linear = np.array([0.0, -9.81, 0.0])  # gravity in -Y

ee_frame = "EndEffector"
ee_id = model.getFrameId(ee_frame)

# --- Desired target in XY plane (reachable!) ---
# Current EE at q=[0,0] is ~[0.43, 0.0], pick target closer to base
x_des = np.array([0.35, 0])  # target X, Y
xdot_des = np.zeros(2)
xddot_des = np.zeros(2)

# PD gains
Kp = np.diag([100, 100])
Kd = np.diag([20, 20])

# --- Dynamics function ---
def robot_dynamics(t, y):
    #y=state vector: [q1,q2,v1,v2]
    #nq = position DOFs
    #nv = velocity DOFs
    q = y[:model.nq] #joint positions
    v = y[model.nq:] #joint velocities

    # Forward kinematics
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    # Jacobian in XY plane
    J = pin.computeFrameJacobian(model, data, q, ee_id)[:2, :]
    allJ=pin.computeJointJacobiansTimeVariation(model, data, q, v)
    J_dot = pin.getFrameJacobianTimeVariation(model, data, ee_id, pin.WORLD)[:2, :]
    # EE position and error in XY plane
    x = data.oMf[ee_id].translation[:2]
    x_err = x_des - x
    xdot_err = xdot_des - J @ v

    # Desired EE acceleration in XY
    x_acc_des = xddot_des + Kd @ xdot_err + Kp @ x_err

    # Inverse dynamics
    B = pin.crba(model, data, q)
    n = pin.rnea(model, data, q, v, np.zeros(model.nv))

    # Map task-space acceleration to joint-space
    qddot_task = np.linalg.pinv(J, rcond=1e-2) @ (x_acc_des - J_dot @ v)  # damping added
    tau = B @ qddot_task + n

    # Compute actual joint accelerations
    qddot = np.linalg.solve(B, tau - n)

    print("Position error XY:", x_err)
    return np.concatenate((v, qddot))

# --- Initial state: slightly bent to avoid singularity ---
y0 = np.array([0.1, -0.1, 0.0, 0.0])  # [q1, q2, v1, v2]

# --- Integrate using RK45 ---
sol = solve_ivp(robot_dynamics, [0, 2.0], y0, method='RK45', max_step=0.001)

# --- Extract joint trajectories ---
q_traj = sol.y[:model.nq, :].T
v_traj = sol.y[model.nq:, :].T

# --- Compute EE trajectory in XY ---
x_traj = []
for q_i in q_traj:
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)
    x_traj.append(data.oMf[ee_id].translation[:2])
x_traj = np.array(x_traj)

print("Simulation done. Final EE XY:", x_traj[-1])
