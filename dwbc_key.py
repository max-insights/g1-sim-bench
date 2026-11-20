"""Send keys to the decoupled-WBC control loop (started with --keyboard-dispatcher-type ros).

    python dwbc_key.py ']'        lower-body RL policy ON        'o'  policy OFF (holds the current joint angles)
    python dwbc_key.py 9          sim only: elastic band on/off  'l'  teleop (arm following) on/off
    python dwbc_key.py w s a d q e z 1 2   keyboard walking / height (ignored while a teleop goal is streaming)
Waits until the control loop's /keyboard_input subscriber is matched, so a key is never lost to discovery.
"""
import sys
import time

import rclpy
from std_msgs.msg import String

keys = sys.argv[1:]
if not keys:
    sys.exit(__doc__)
rclpy.init()
node = rclpy.create_node("dwbc_key")
pub = node.create_publisher(String, "/keyboard_input", 10)
t0 = time.monotonic()
while pub.get_subscription_count() == 0:
    if time.monotonic() - t0 > 5:
        print("no /keyboard_input subscriber within 5 s (control loop not running, or another ROS_DOMAIN_ID)")
        sys.exit(1)
    time.sleep(0.05)
time.sleep(0.2)
for k in keys:
    m = String()
    m.data = k
    pub.publish(m)
    print(f"sent {k!r}", flush=True)
    time.sleep(0.15)
time.sleep(0.3)
node.destroy_node()
rclpy.shutdown()
