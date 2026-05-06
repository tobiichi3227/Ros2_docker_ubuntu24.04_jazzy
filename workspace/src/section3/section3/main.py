import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import threading
import sys
import termios
import tty
import select
from rclpy.executors import SingleThreadedExecutor
import onnxruntime as ort
import numpy as np
import joblib
from geometry_msgs.msg import Point
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
class State(Node):
    def __init__(self):
        super().__init__('state')

        model_path = "/workspace/src/model.pkl"
        self.x_boundary_min = -0.1494
        self.x_boundary_max =  0.1494
        self.z_boundary_min = -0.1182
        self.z_boundary_max =  0.1182
        self.ball_radius = 0.02   
        self.bottom_board_y = 0.042
        self.top_board_y = 0.312
        self.model = joblib.load(model_path)
        self.sess = ort.InferenceSession("/workspace/src/model.onnx")
        self.mode = None
        self.ball_p = [0.0, 0.0, 0.0]
        self.ball_v = [0.0, 0.0, 0.0]
        self.goal_ball = [0.0, 0.0, 0.0]
        self.s1 = self.s2 = self.s3 = self.s4 = self.s5 = self.s6 = 0.0
        self.goal_s1 = self.goal_s2 = self.goal_s3 = self.goal_s4 = self.goal_s5 = self.goal_s6 = 0.0

        self.keyboard_thread = None
        self.running = False
        self.debug_timer = None
        self.ai_control_timer = None  
        self.exit_requested = False


        self.create_subscription(Float32, '/ball_vx', self.vx_cb, 10)
        self.create_subscription(Float32, '/ball_vy', self.vy_cb, 10)
        self.create_subscription(Float32, '/ball_vz', self.vz_cb, 10)
        self.create_subscription(Point, '/ball_position', self.ball_cb, 10)
        self.create_subscription(Float32, '/down_plus', self.s1_cb, 10)
        self.create_subscription(Float32, '/down_min', self.s2_cb, 10)
        self.create_subscription(Float32, '/down_minus', self.s3_cb, 10)
        self.create_subscription(Float32, '/up_plus', self.s4_cb, 10)
        self.create_subscription(Float32, '/up_min', self.s5_cb, 10)
        self.create_subscription(Float32, '/up_minus', self.s6_cb, 10)

        self.goal_pubs = {
            'goal_s1': self.create_publisher(Float32, '/goal_s1', 10),
            'goal_s2': self.create_publisher(Float32, '/goal_s2', 10),
            'goal_s3': self.create_publisher(Float32, '/goal_s3', 10),
            'goal_s4': self.create_publisher(Float32, '/goal_s4', 10),
            'goal_s5': self.create_publisher(Float32, '/goal_s5', 10),
            'goal_s6': self.create_publisher(Float32, '/goal_s6', 10),
        }

        self.debug_timer = self.create_timer(1.0, self.debug_print)

    # ----- callbacks -----
    def ball_cb(self, msg: Point):
        self.ball_p = [msg.x, msg.y, msg.z]
    def vx_cb(self, msg): self.ball_v[0] = msg.data
    def vy_cb(self, msg): self.ball_v[1] = msg.data
    def vz_cb(self, msg): self.ball_v[2] = msg.data
    def s1_cb(self, msg): self.s1 = msg.data
    def s2_cb(self, msg): self.s2 = msg.data
    def s3_cb(self, msg): self.s3 = msg.data
    def s4_cb(self, msg): self.s4 = msg.data
    def s5_cb(self, msg): self.s5 = msg.data
    def s6_cb(self, msg): self.s6 = msg.data

    def set_mode(self, mode: str):
        self.mode = mode
        self.get_logger().info(f"模式: {mode.upper()}")

     
        if self.ai_control_timer is not None:
            self.ai_control_timer.cancel()
            self.ai_control_timer = None

        if mode == 'manual':
            if self.debug_timer:
                self.debug_timer.cancel()
                self.debug_timer = None
            self.get_logger().info("keyboard：")
            self.get_logger().info("  q/w → goal_s1  +/-   |  e/r → goal_s2  +/-")
            self.get_logger().info("  t/y → goal_s3  +/-   |  u/i → goal_s4  +/-")
            self.get_logger().info("  o/p → goal_s5  +/-   |  a/s → goal_s6  +/-")
            self.get_logger().info("  ESC → exit manual mode")
            self.start_keyboard_listener()

        elif mode == 'ai':
            if self.debug_timer is None:
                self.debug_timer = self.create_timer(1.0, self.debug_print)
                
   
            for i in range(3):
                if self.ball_v[i] > 0:
                    self.ball_v[i] = 1.0
                elif self.ball_v[i] < 0:
                    self.ball_v[i] = -1.0
            self.ai_control_timer = self.create_timer(0.001, self.update_ai_control)

        elif mode == 'math':
            if self.debug_timer is None:
                self.debug_timer = self.create_timer(1.0, self.debug_print)
            self.ai_control_timer = self.create_timer(0.001, self.update_math_control)

    def update_ai_control(self):
        inputs = {
            "x": np.array([[self.ball_p[0]]], np.float64),
            "y": np.array([[self.ball_p[1]]], np.float64),
            "z": np.array([[self.ball_p[2]]], np.float64),
            "vx": np.array([[self.ball_v[0]]], np.float64),
            "vy": np.array([[self.ball_v[1]]], np.float64),
            "vz": np.array([[self.ball_v[2]]], np.float64),
        }
        pz = float(self.sess.run(None, inputs)[0][0][0])
        if self.ball_v[1] > 0:
            self.publish_goal('goal_s4', -pz)
            self.publish_goal('goal_s5', -pz)
            self.publish_goal('goal_s6', -pz)
        else:
            self.publish_goal('goal_s1', pz)
            self.publish_goal('goal_s2', pz)
            self.publish_goal('goal_s3', pz)

    def predict_landing_position(self, posX, posY, posZ, velX, velY, velZ):
        def binarize(v):
            return 1 if v > 0 else -1
        vx, vy, vz = map(binarize, [velX, velY, velZ])
        dir_class = self.encode_direction(vx, vy, vz)
        input_array = np.array([[posX, posY, posZ, dir_class]])
        prediction = self.model.predict(input_array)
        return (prediction[0,0], prediction[0,1], prediction[0,2])

    def encode_direction(self, vx, vy, vz):
        if vx == 1 and vy == 1 and vz == 1: return 1
        elif vx == 1 and vy == 1 and vz == -1: return 2
        elif vx == 1 and vy == -1 and vz == 1: return 3
        elif vx == -1 and vy == 1 and vz == 1: return 4
        elif vx == -1 and vy == -1 and vz == 1: return 5
        elif vx == -1 and vy == 1 and vz == -1: return 6
        elif vx == 1 and vy == -1 and vz == -1: return 7
        elif vx == -1 and vy == -1 and vz == -1: return 8
        else: return 0


    def update_math_control(self):

        if self.ball_v[1] > 0:
            target_y = self.top_board_y
        elif self.ball_v[1] < 0:
            target_y = self.bottom_board_y
        else:
            return  

        result = self.predict_landing_with_walls_and_radius(
            self.ball_p[0], self.ball_p[1], self.ball_p[2],
            self.ball_v[0], self.ball_v[1], self.ball_v[2],
            target_y,
            self.x_boundary_min, self.x_boundary_max,
            self.z_boundary_min, self.z_boundary_max,
            self.ball_radius
        )

        if result[0] is None:
            self.get_logger().warn(f"none(target_y={target_y})")
            return

        self.goal_ball = result
        self.get_logger().info(f"Math: x={result[0]:.3f}, z={result[2]:.3f}")
        self.auto_go_to_goal()

    def predict_landing_with_walls_and_radius(self, posX, posY, posZ, velX, velY, velZ, target_y,
                                              x_boundary_min, x_boundary_max,
                                              z_boundary_min, z_boundary_max,
                                              ball_radius=0.02):
      
        if abs(velY) < 1e-6:
            return (None, None, None)
        t = (target_y - posY) / velY
        if t <= 0:
            return (None, None, None)

        x_min_eff = x_boundary_min + ball_radius
        x_max_eff = x_boundary_max - ball_radius
        z_min_eff = z_boundary_min + ball_radius
        z_max_eff = z_boundary_max - ball_radius

        def reflect_1d(p, v, low, high, dt):
            span = high - low
            if span <= 0:
                return p + v * dt
            p_abs = p + v * dt
            offset = p_abs - low
      
            if offset >= 0:
                cycles = int(offset / span)
            else:
           
                cycles = -int((-offset) / span) - 1 if (offset % span) != 0 else -int((-offset) / span)
            remainder = offset - cycles * span
            if cycles % 2 == 0:
                return low + remainder
            else:
                return high - remainder

        landX = reflect_1d(posX, velX, x_min_eff, x_max_eff, t)
        landZ = reflect_1d(posZ, velZ, z_min_eff, z_max_eff, t)
        return (landX, target_y, landZ)

    def start_keyboard_listener(self):
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            return
        self.running = True
        self.exit_requested = False
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.keyboard_thread.start()

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            print("\r", end='', flush=True)
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    self._handle_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print("\r\n", end='', flush=True)

    def _handle_key(self, key: str):
        step = 0.1
        mapping = {
            'q': ('goal_s1', +step), 'w': ('goal_s1', -step),
            'e': ('goal_s2', +step), 'r': ('goal_s2', -step),
            't': ('goal_s3', +step), 'y': ('goal_s3', -step),
            'u': ('goal_s4', +step), 'i': ('goal_s4', -step),
            'o': ('goal_s5', +step), 'p': ('goal_s5', -step),
            'a': ('goal_s6', +step), 's': ('goal_s6', -step),
        }
        if key == '\x1b':
            self.running = False
            self.exit_requested = True
            return
        if key in mapping:
            attr, delta = mapping[key]
            new_val = getattr(self, attr) + delta
            setattr(self, attr, new_val)
            self.publish_goal(attr, new_val)
            robot_attr = attr.replace('goal_', '')
            robot_val = getattr(self, robot_attr)
            print(f"\r{attr} = {new_val:.3f} | {robot_attr} = {robot_val:.3f}  ", end='', flush=True)

    def stop_keyboard_listener(self):
        self.running = False
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1.0)

    def debug_print(self):
        if self.mode == 'manual':
            return
       
        self.get_logger().info(
            f"input: px={self.ball_p[0]:.5f}, py={self.ball_p[1]:.5f}, pz={self.ball_p[2]:.5f}, "
            f"vx={self.ball_v[0]:.5f}, vy={self.ball_v[1]:.5f}, vz={self.ball_v[2]:.5f}"
        )
        self.get_logger().info(f"predix: x={self.goal_ball[0]:.5f}, y={self.goal_ball[1]:.5f}, z={self.goal_ball[2]:.5f}")

    def publish_goal(self, goal_name: str, value: float):
        if goal_name in self.goal_pubs:
            msg = Float32()
            msg.data = value
            self.goal_pubs[goal_name].publish(msg)

    def auto_go_to_goal(self):
        x, y, z = self.goal_ball
        goal_1 = z
        goal_2 = -z
        if y > 0.15:
            if x > 0.06:
                self.goal_s4 = goal_1
            elif x < -0.06:      
                self.goal_s5 = goal_1
            else:
                self.goal_s6 = goal_1
        elif y < 0.15:
            if x > 0.06:
                self.goal_s1 = goal_2
            elif x < -0.06:
                self.goal_s2 = goal_2
            else:
                self.goal_s3 = goal_2
        for name in ['goal_s1','goal_s2','goal_s3','goal_s4','goal_s5','goal_s6']:
            self.publish_goal(name, getattr(self, name))

def main():
    rclpy.init()
    while True:
        node = State()
        while True:
            choice = input("press a to AI mode, press m to manual mode, press mm to math mode: ").strip().lower()
            if choice == 'a':
                node.set_mode('ai')
                break
            elif choice == 'm':
                node.set_mode('manual')
                break
            elif choice == 'mm':
                node.set_mode('math')
                break
            else:
                print("not accept (a / m / mm)")

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            while rclpy.ok() and not node.exit_requested:
                executor.spin_once(timeout_sec=0.1)
        except KeyboardInterrupt:
            print("\n unconnect")
            break
        finally:
            node.stop_keyboard_listener()
            executor.shutdown()
            node.destroy_node()
        if node.exit_requested:
            continue
        else:
            break
    rclpy.shutdown()

if __name__ == '__main__':
    main()
