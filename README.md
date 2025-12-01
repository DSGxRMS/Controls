# Control Algorithm

This repository contains a working control algorithm implementation. While functional, it is currently in an unoptimized state and under active development.

## Prerequisites

- The simulator should be opened in acceleration mode, though it is also compatible with velocity mode.

## Setup

After cloning this repository, run the following commands to build and set up the environment:

```bash
colcon build 
source ./install/setup.bash
```

## Usage

### Running the Control Algorithm

To execute the control algorithm:

# For Older version

```bash
ros2 control run_control
```

# For Version 2 working with Simulated PathPlanning

First setup the path points publishing node 
```bash
ros2 control_v2 pp_publisher 
```

Then to run the control logic loop run 
- ⚠️ Only tested for velocity mode 
```bash
ros2 run control_v2 control_loop
```

### Testing

To verify the control functionality:

```bash
ros2 control test_control
```