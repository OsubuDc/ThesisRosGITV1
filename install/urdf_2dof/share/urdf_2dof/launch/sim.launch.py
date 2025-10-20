import os
import launch
from launch.substitutions import LaunchConfiguration
import launch_ros
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package path
    pkg_path = get_package_share_directory('urdf_2dof')
    
    # Construct file paths
    urdf_model_path = os.path.join(pkg_path, 'urdf', '2dof.urdf')
    mesh_dir = os.path.join(pkg_path, 'meshes')
    rviz_config_path = os.path.join(pkg_path, 'config', 'config.rviz')
    
    print('URDF model path =', urdf_model_path)
    
    # Load URDF content
    with open(urdf_model_path, 'r') as infp:
        robot_desc = infp.read()
    
    params = {'robot_description': robot_desc}
    
    # Define Nodes to launch
    # Node 1: Robot State Publisher
        # 1->node start 2->looks for robot_description parameter 3->build kinematic tree 4-> subscribe to joint states 5->publish TF transforms
        # everytime new joint state message arrives it update: joint positions, computes transforms, publishes transforms to /tf topic.
    # Publishes TF transforms based on the URDF and joint states. /tf topic and /tf_static topic.
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',    # ROS package name (standard node)
        executable='robot_state_publisher', # Node executable name
        output='screen',                    # Print output to terminal
        parameters=[params]                 # Pass robot_description as parameter  
    )
    
    # Node 2: Your Pendulum Simulator
    # Publishes joint_states based on dynamics
    pendulum_sim_node = launch_ros.actions.Node(
        package='urdf_2dof',
        executable='pendulum_sim_node.py',
        name='pendulum_simulator',
        output='screen',
        parameters=[
            {'urdf_path': urdf_model_path},
            {'mesh_dir': mesh_dir}
        ]
    )
    
    # Node 3: RViz2
    # Opens RViz and automatically loads config file.
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )
    
    return launch.LaunchDescription([
        robot_state_publisher_node,
        pendulum_sim_node,
        rviz_node
    ])
