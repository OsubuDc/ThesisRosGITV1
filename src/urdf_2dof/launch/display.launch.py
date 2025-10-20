import os
import launch
from launch.substitutions import LaunchConfiguration
import launch_ros


def generate_launch_description():
    # Get paths to package and URDF/RViz files
    # Find the absolute path to package 'urdf_2dof'
    pkg_path = launch_ros.substitutions.FindPackageShare(package='urdf_2dof').find('urdf_2dof')

    # Construct absolute file paths to URDF and RViz config
    urdf_model_path = os.path.join(pkg_path, 'urdf/2dof.urdf')
    rviz_config_path = os.path.join(pkg_path, 'config/config.rviz')

    # Shows path in terminal when launching
    print('URDF model path =', urdf_model_path)

    # Load URDF file content into a string
    # ROS 2 expects the robot_description parameter as a single string
    with open(urdf_model_path, 'r') as infp: #open() is a built-in function in python, r -> read mode, as infp -> creates file object that you can read from.
        robot_desc = infp.read() #.read() reads the entire file content and stores it in robot_desc variable as one big string.

    # Store parameters in a dictionary
    params = {'robot_description': robot_desc} #python dictionary with one key-value pair. ROS2 expects parameters as dictionaries.

# Define Nodes to launch
    # Node 1: Robot State Publisher
        # 1->node start 2->looks for robot_description parameter 3->build kinematic tree 4-> subscribe to joint states 5->publish TF transforms
        # everytime new joint state message arrives it update: joint positions, computes transforms, publishes transforms to /tf topic.
    # Publishes TF transforms based on the URDF and joint states. /tf topic and /tf_static topic.
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',     # ROS package name (standard node)
        executable='robot_state_publisher',  # Node executable name
        output='screen',                     # Print output to terminal
        parameters=[params]                  # Pass robot_description as parameter
    )
# for pendulum simulator neither is used. Node 2 is fixed representation, node 3 is interactive representation.
    # Node 2: Joint State Publisher (non-GUI)
    # Publishes joint angles for each joint, useful for headless mode.
    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[params],
        condition=launch.conditions.UnlessCondition(LaunchConfiguration('gui'))
        # This condition runs this node only when gui:=False
    )

    # Node 3: Joint State Publisher GUI
    # Allows you to move the joints with sliders in a GUI.
    joint_state_publisher_gui_node = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=launch.conditions.IfCondition(LaunchConfiguration('gui'))
        # This condition runs this node only when gui:=True
    )

    # Node 4: RViz2
    # Opens RViz and automatically loads config file.
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]  # Load specific RViz configuration. -d flag tells RViz to automatically load the specified config file on startup.
    )

    # Declare Launch Arguments
    # Launch arguments can be passed via command line:
    # Example: ros2 launch urdf_2dof display.launch.py gui:=False
    gui_arg = launch.actions.DeclareLaunchArgument(
        name='gui',
        default_value='True',
        description='Flag to enable joint_state_publisher_gui'
    )

    model_arg = launch.actions.DeclareLaunchArgument(
        name='model',
        default_value=urdf_model_path,
        description='Absolute path to the robot URDF file'
    )

    # Return all launch actions together
    return launch.LaunchDescription([
        gui_arg,
        model_arg,
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])
 