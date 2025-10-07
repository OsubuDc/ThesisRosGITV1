import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Path to your URDF and meshes folder
urdf_path = "/ThesisRosGITV1/V2_URDF/urdf/V2_URDF_fixed.urdf"
mesh_dir = "/ThesisRosGITV1/V2_URDF/meshes"

# Load robot
robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
model = robot.model
data = robot.data

print("Robot model loaded successfully.", model.name)
