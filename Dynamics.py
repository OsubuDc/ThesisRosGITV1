import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.integrate import solve_ivp
import sys


# --- Load robot model ---
urdf_path = "/ThesisRosGITV1/V2_URDF/urdf/V2_URDF_fixed.urdf"
mesh_dir = "/ThesisRosGITV1/V2_URDF/meshes"
robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
model, data = robot.model, robot.data
model.gravity.linear = np.array([0.0, -9.81, 0.0])  # gravity in -Y

ee_frame = "EndEffector"
ee_id = model.getFrameId(ee_frame)

# --- Desired target in XY plane (reachable!) ---
x_des = np.array([0.05, 0.05])
xdot_des = np.zeros(2)
xddot_des = np.zeros(2)

# PD gains
Kp = np.diag([100, 100])
Kd = np.diag([20, 20])

# --- Dynamics function ---
def robot_dynamics(t, y):
    q = y[:model.nq]
    v = y[model.nq:]

    # Forward kinematics
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    # Jacobian and its derivative
    J = pin.computeFrameJacobian(model, data, q, ee_id)[:2, :]
    J_dot = pin.getFrameJacobianTimeVariation(model, data, ee_id, pin.WORLD)[:2, :]

    # EE position and errors
    x = data.oMf[ee_id].translation[:2]
    x_err = x_des - x
    xdot_err = xdot_des - J @ v

    # Desired EE acceleration
    x_acc_des = xddot_des + Kd @ xdot_err + Kp @ x_err

    # Inverse dynamics
    B = pin.crba(model, data, q)
    n = pin.rnea(model, data, q, v, np.zeros(model.nv))

    # Map task-space acceleration to joint-space
    qddot_task = np.linalg.pinv(J, rcond=1e-2) @ (x_acc_des - J_dot @ v)
    u = B @ qddot_task + n

    # Actual joint accelerations
    qddot = np.linalg.solve(B, u - n)

    # --- Live print (updates one line) ---
    sys.stdout.write(
        f"\rTime: {t:5.3f} s | EE XY: [{x[0]:+.3f}, {x[1]:+.3f}] | "
        f"Err: [{x_err[0]:+.4f}, {x_err[1]:+.4f}]"
    )
    sys.stdout.flush()

    return np.concatenate((v, qddot))

# --- Initial state: slightly bent to avoid singularity ---
y0 = np.array([0.2, 0.1, 0.0, 0.0])

# --- Integrate using RK45 ---
print("Running simulation...")
sol = solve_ivp(robot_dynamics, [0, 4.0], y0, method='RK45', max_step=0.001)
print("\nSimulation finished.\n")

# --- Extract trajectories ---
q_traj = sol.y[:model.nq, :].T
v_traj = sol.y[model.nq:, :].T

# --- Compute EE trajectory in XY ---
x_traj = []
for q_i in q_traj:
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)
    x_traj.append(data.oMf[ee_id].translation[:2])
x_traj = np.array(x_traj)

print(f"Final EE XY: {x_traj[-1]}")
