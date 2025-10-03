# Improved ROS2 Control System

This package provides an enhanced vehicle control system for path following with robust error handling, startup assistance, and comprehensive debugging capabilities.

## Key Improvements

### ✅ Fixed Issues
- **Vehicle Movement**: Added timer-based publishing and startup assistance
- **Path Coordinates**: Fixed coordinate transformation with configurable offsets
- **Velocity Constraints**: Reduced minimum velocity from 5.0 m/s to 0.5 m/s for gentle startup
- **Error Handling**: Added comprehensive exception handling and validation
- **Debugging**: Added extensive logging and monitoring capabilities

### 🆕 New Features
- **Startup Mode**: Automatic assistance to get vehicle moving from standstill
- **Timer-based Publishing**: Consistent control output regardless of odometry availability
- **Configurable Path Offsets**: Align path with vehicle starting position
- **Debug Logging**: Monitor control commands and system state
- **Test Framework**: Built-in testing utilities

## Usage

### Basic Usage
```bash
# Run with default parameters
ros2 run control run_control

# Run with custom parameters
ros2 run control run_control --ros-args \
  -p odom_topic:=/ground_truth/odom \
  -p cmd_topic:=/cmd \
  -p mode:=ackermann \
  -p hz:=50.0 \
  -p enable_startup_mode:=true
```

### Using Launch File
```bash
# Basic launch
ros2 run control launch_control.py

# Launch with custom parameters
ros2 run control launch_control.py \
  odom_topic:=/ground_truth/odom \
  cmd_topic:=/cmd \
  debug:=true
```

### Testing the System
```bash
# Test with stationary vehicle (default)
ros2 run control test_control.py

# Test with moving vehicle
ros2 run control test_control.py --ros-args -p test_mode:=moving

# Test with circular path
ros2 run control test_control.py --ros-args -p test_mode:=path
```

## Configuration Parameters

### Core Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom_topic` | `/odom` | Odometry topic name |
| `cmd_topic` | `/cmd` | Command output topic |
| `mode` | `ackermann` | Control mode (`ackermann` or `twist`) |
| `hz` | `50.0` | Control loop frequency (Hz) |

### Path Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `path_csv` | `pathpoints.csv` | Path to waypoint CSV file |
| `scaling_factor` | `1.0` | Scale factor for path coordinates |
| `loop` | `false` | Whether path is a closed loop |
| `path_offset_x` | `0.0` | Path offset in X direction |
| `path_offset_y` | `0.0` | Path offset in Y direction |

### Control Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_startup_mode` | `true` | Enable startup assistance |
| `qos_best_effort` | `true` | Use BEST_EFFORT QoS |

## Troubleshooting

### Vehicle Not Moving
1. **Check Odometry**: Ensure odometry data is being published
   ```bash
   ros2 topic echo /odom
   ```

2. **Check Control Commands**: Monitor output commands
   ```bash
   ros2 topic echo /cmd
   ```

3. **Enable Debug Logging**: Run with debug output
   ```bash
   ros2 run control run_control --ros-args --log-level DEBUG
   ```

4. **Test with Test Script**: Use the test framework
   ```bash
   ros2 run control test_control.py
   ```

### Path Issues
1. **Check Path Loading**: Look for path loading messages in logs
2. **Adjust Path Offset**: Use `path_offset_x` and `path_offset_y` parameters
3. **Verify CSV Format**: Ensure CSV has `x,y` columns

### Control Issues
1. **Check Topic Names**: Verify topic names match your simulator
2. **Try Different Mode**: Switch between `ackermann` and `twist` modes
3. **Adjust Control Rate**: Modify `hz` parameter

## Code Structure

```
control/
├── ros_bag_control.py     # Main control node
├── controls_functions.py  # Control algorithms and utilities
├── test_control.py        # Testing framework
├── launch_control.py      # Launch configuration
├── pathpoints.csv         # Default path waypoints
└── README.md             # This file
```

### Key Classes
- `ControlNode`: Main ROS2 node handling control logic
- `ControlAlgorithm`: Path following algorithms (Pure Pursuit, PID)
- `ControlTester`: Test framework for validation

## Control Algorithm Details

### Startup Mode
When `enable_startup_mode=true`:
1. Vehicle starts with gentle throttle (0.3) until reaching minimum speed
2. Gradually increases throttle based on speed error
3. Switches to normal path following once startup speed is reached

### Path Following
1. **Pure Pursuit**: Generates steering commands based on lookahead point
2. **PID Control**: Longitudinal control for speed tracking
3. **Lateral Correction**: Additional PID for cross-track error correction

### Safety Features
- Zero command fallback when no odometry available
- Exception handling for all critical functions
- Parameter validation and bounds checking
- Graceful degradation on errors

## Advanced Usage

### Custom Path Creation
Create CSV file with `x,y` columns:
```csv
x,y
0.0,0.0
1.0,0.0
2.0,1.0
3.0,1.0
```

### Parameter Tuning
Key parameters to adjust:
- `STARTUP_THROTTLE`: Initial throttle for startup (in controls_functions.py)
- `V_MIN`: Minimum target velocity
- PID gains in `ControlAlgorithm` class

### Integration with Simulators
Common topic mappings:
- **Gazebo**: `/odom` → `/cmd`
- **AirSim**: `/airsim_node/drone_1/odom_local_ned` → `/airsim_node/drone_1/car_cmd`
- **CARLA**: `/carla/ego_vehicle/odometry` → `/carla/ego_vehicle/ackermann_cmd`

## Support

For issues or questions:
1. Check logs with debug mode enabled
2. Use the test framework to isolate problems
3. Verify topic names and message types match your setup
4. Ensure path coordinates align with vehicle starting position