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

```bash
ros2 control run_control
```

### Testing

To verify the control functionality:

```bash
ros2 control test_control
```