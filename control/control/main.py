import rclpy
from .ros_interface import ControlNode

def main(args=None):
    """
    Main entry point for the control node.
    """
    rclpy.init(args=args)
    
    control_node = ControlNode()
    
    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    finally:
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()