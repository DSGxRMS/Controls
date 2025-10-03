import rclpy
from .ros_interface import RosInterfaceNode

def main():
    rclpy.init()
    node = RosInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()