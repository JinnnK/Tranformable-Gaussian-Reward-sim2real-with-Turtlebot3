# Tranformable-Gaussian-Reward-sim2real-with-Turtlebot3
This repository contains the sim2real procedure and code for our paper titled ["Transformable Gaussian Reward Function for Robot Navigation with Deep Reinforcement Learning"](https://github.com/JinnnK/Transformable-Gaussian-Reward-for-Robot-Nav).
In sim2real, we adapted a people detector and SLAM from previous works, and transfered a simulated crowd navigation policy to a TurtleBot3 without any real-world training.
For more details, please refer to the [Notion]() and [arXiv preprint]().
For experiment demonstrations, please refer to the [youtube video]().
This repo only serves as a reference point for the sim2real transfer of crowd navigation. And this repository is a modification of the sim2real method using TurtleBot2i to apply it to TurtleBot3. Therefore, source code contains many parts where the term "turtlebot2i" is used. However, it works with TurtleBot3. If you wish to use TurtleBot2i, please refer to the following [GitHub repository](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot).
Since there are lots of uncertainties in real-world experiments that may affect performance, **we cannot guarantee that it is reproducible on your case.**

## overview
### Hardware
- Host computer:
  - CPU: Intel i5-10600 @ 3.30GHz
  - GPU: Nvidia RTX 3070
  - Memory: 48GB
- Turtlebot3:
  - On-board computer: OpenCR
  - Lidar: LDS-01
  - Mobile base: Turtlebot3
### Software
The host computer and the turtlebot communicates through ROS by connecting to the same WiFi network.
- Host computer:
  - OS: Ubuntu 20.04
  - Python version: 3.9.16
  - Cuda version: 11.4
  - ROS version: Noetic **(our code WILL NOT WORK with lower versions of ROS on host computer)**
- Turtlebot3:
  - OS: Linux
  - ROS version: Kinetic


## Setup

### Turtlebot3
1. Create a catkin workspace
```
mkdir ~/catkin_ws
cd catkin_ws
mkdir -p src
catkin_make
cd src
```

2. Install ROS packages into your workspace
```
cd ~/catkin_ws/src
# turtlebot3
sudo apt install ros-noetic-dynamixel-sdk
sudo apt install ros-noetic-turtlebot3-msgs
sudo apt install ros-noetic-turtlebot3

cd ~/catkin_ws
catkin_make
```
For more details, please refer to [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/).

### Host computer
1. Create a catkin workspace
```
mkdir ~/catkin_ws
cd catkin_ws
mkdir -p src
catkin_make
cd src
```

2-1. Install ROS packages into your workspace
```
cd ~/catkin_ws/src

# people detector
git clone https://github.com/VisualComputingInstitute/2D_lidar_person_detection.git
cd 2D_lidar_person_detection/dr_spaam
python setup.py install
cd ~/catkin_ws
catkin_make
catkin build dr_spaam_ros
```
2-2. Download datasets and modify the configs
You can download the dataset from [here](https://drive.google.com/drive/folders/1Wl2nC8lJ6s9NI1xtWwmxeAUnuxDiiM4W). And in the `topics.yaml` located in `dr_spaam_ros/config/`, change the subscriber from `/segway/scan_multi` to `/scan`. Additionally, in the `dr_spaam_ros.yaml`, modify the weight_file path and detector_model according to the pre-trained model you want to use.

Please note that 2D_lidar_person_detection works with numpy version 1.23.0 or lower.
For more details, please refer to [here](https://github.com/VisualComputingInstitute/2D_lidar_person_detection).

## Run the code
### Training in simulation
- Clone the [navigation repo](https://github.com/JinnnK/Transformable-Gaussian-Reward-for-Robot-Nav)
- Modify the configurations.
  1. Modify the configurations in `crowd_nav/configs/config.py` and `arguments.py` following instructions [here](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph#training)

  2. In `crowd_nav/configs/config.py`, set
     - `action_space.kinematics = "unicycle"` if your robot has a differential drive
       - Explanations on [holonomic robot](https://techtv.mit.edu/videos/9392041131714766aa78ffadf4e8c76c/#:~:text=Holonomic%20drive%2C%20in%20the%20realm,direction%2C%20would%20not%20be%20holonomic.) v.s. [differential drive robot](https://msl.cs.uiuc.edu/planning/node659.html)
     - adjust `sim.circle_radius`, `sim.arena_size`, and `sim.human_num` based on your real environment
  3. In the `arguments.py`, set `use-linear-lr-decay` to True on line 124.
  4. In the `crowd_sim/envs/ros_turtlebot2i_env.py`, replace `/cmd_vel` on line 80 with the appropriate command node's name.
  5. Replace the `dr_spaam_ros.py` located at `2D_lidar_person_detection/dr_spaam_ros/src/dr_spaam_ros` with the `dr_spaam_ros.py` file uploaded in this source code.
  6. Open the `config.py` file located in your project's directory. Set `sim2real.use_dummy_detect` to `False`.
  7. In the `crowd_sim/envs/ros_turtlebot2i_env.py`, In the `state_cb` function, add the following lines:
   
   ```python
   humanArrayMsg.header.frame_id = "map"
   jointStateMsg.header.frame_id = "map"
   ```
  8. Add `O['visible_masks'][:] = True` to the `process_obs_rew` function in `Crowd_Nav_Attn/rl/vec_env/vec_pretext_normalize.py`.
  9. In `crowd_sim/envs/ros_turtlebot2i_env.py`, adjust the value of the `beta` parameter in the `smooth` function for fine tuning. Please note that incorrect tuning might lead to discrepancies between simulation and actual movement.

- After you change the configurations, run
  ```
  python train.py
  ```
- The checkpoints and configuration files will be saved to the folder specified by `output_dir` in `arguments.py`.

- Test the trained policy in simulation following instructions, make sure the results are satisfactory (success rate is at least around 90%)

### Testing in real world
1. Create a map of the real environment using SLAM:
   - **[Host computer]** Run roscore
      ```
      roscore
      ```
   - **[Turtlebot]** Launch the mobile base:
      ```
      ssh pi@{IP_ADDRESS_OF_TURTLEBOT3}
      roslaunch turtlebot3_bringup turtlebot3_robot.launch
      ```
   - **[Host computer]** Launch SLAM and navigation
      ```
      roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
      ```
   - **[Host computer]** Launch robot teleoperation
      ```
      roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
      ```
   - **[Host computer]** Teleoperate the robot around the environment until you are satisfied with the map in rviz, save the map by
      ```
      rosrun map_server map_saver -f ~/map
      ```
      In your home directory, you will see two files: `map.yaml` and `map.pgm`.

2. Then, test the trained policy in real turtlebot in the mapped environment:
   - **[Turtlebot]** Launch the mobile base (skip this if it is running)
      ```
      roslaunch turtlebot3_bringup turtlebot3_robot.launch
      ```
   - **[Host computer]** Load the map
     ```
     rosrun map_server map_server map.yaml
     ```
   - **[Host computer]** Launch navigation
     ```
     roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml
     ```
   - **[Host computer]** Run the DR-SPAAM people detector:
     ```
     roslaunch dr_spaam_ros dr_spaam_ros.launch
     ```
   - **[Host computer]** Launch rviz to watch the 2D_lidar_person_detection:
     ```
     rosrun rviz rviz -d /home/hri/catkin_ws/src/2D_lidar_person_detection/dr_spaam_ros/example.rviz
     ```
     In the rviz, change `/segway/scan_multi` to `/scan`.
   - **[Host computer]** cd into the crowd navigation repo,
     - in `trained_models/your_output_dir/arguments.py`, change `env-name` to `'rosTurtlebot2iEnv-v0'`
     - in `trained_models/your_output_dir/configs/config.py`, change configurations under `sim2real` if needed
     - then run
       ```
       python test.py
       ```
     Type in the goal position following the terminal output, and the robot will execute the policy if everything works.

## Disclaimer
1. We only tested our code in the above listed hardware and software settings. It may work with other robots/versions of software, but we do not have any guarantee.

2. If the RL training does not converge, we recommend starting with an easier setting (fewer humans, larger circle radius, larger robot speed, etc) and then gradually increase the task difficulty during training.

3. The performance of our code can vary depending on the choice of hyperparameters and random seeds (see [this reddit post](https://www.reddit.com/r/MachineLearning/comments/rkewa3/d_what_are_your_machine_learning_superstitions/)).
Unfortunately, we do not have time or resources for a thorough hyperparameter search. Thus, if your results are slightly worse than what is claimed in the paper, it is normal.
To achieve the best performance, we recommend some manual hyperparameter tuning.


## Citation
If you find the code or the paper useful for your research, please cite the following papers:
```
@inproceedings{,
  title={},
  author={Jinyeob Kim},
  booktitle={},
  year={2023}
}

@inproceedings{liu2022intention,
  title={Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author={Liu, Shuijing and Chang, Peixin and Huang, Zhe and Chakraborty, Neeloy and Hong, Kaiwen and Liang, Weihang and Livingston McPherson, D. and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```

## Credits
[Jinyeob Kim](https://github.com/JinnnK)

Part of the code is based on the following repositories:

[1] P. Chang, N. Chakraborty, and Z. Huang, "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph," in IEEE International Conference on Robotics and Automation (ICRA), 2023. (Github: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)

[2] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell, "Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning," in IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 3517-3524. (Github: https://github.com/Shuijing725/CrowdNav_DSRNN)

[3] Z. Huang, R. Li, K. Shin, and K. Driggs-Campbell. "Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction," in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1198–1205, 2022. (Github: https://github.com/tedhuang96/gst)


## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.
