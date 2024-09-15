# Model Predictive Control and Trajectory Planning for Quadrupedal Locomotion

This repository contains solution to the final assignments for the Legged Robots lecture (CIT436000) at TU Munich. 

This project focuses on the development and evaluation of a motion planning and control system for legged robots, specifically addressing the challenges of generating stable and dynamic locomotion. The project leverages a combination of Model Predictive Control (MPC) and trajectory planning to achieve coordinated movements between the robot's base body and its individual legs. The framework has been implemented within the MuJoCo simulation environment and simulated for the robot
[Go2
Quadruped Robot](https://www.unitree.com/go2/) by [Unitree
Robotics](https://www.unitree.com/).

The detailed report for the project can be found in the [`final_project.ipynb`](final_project.ipynb). 

## Installation

1. Clone this repository:

2. Create and activate a Conda environment from the provided YAML file:
    ```bash
    conda env create -f LeggedRobotsGP.yml
    conda activate LeggedRobotsGP
    ```

3. Ensure MuJoCo is installed on your system, and set up your environment to include the MuJoCo Python bindings.


```bash              
├── model/                       # Directory for robot model files (Unitree Go2)
├── project_code/                # Main project code directory
│   ├── controller/              # Robot joints controller
│   ├── planner/                 # Full body and foot position planners
│   ├── state/                   # State management for the robot
│   ├── utils/
│   └── constants.py             # Constants for the project
├── .gitignore             
├── final_project.ipynb          # Complete project demonstration aka the report
├── LeggedRobotsGP.yml           
├── MPC_Example.ipynb            # MPC use case demonstrations
```