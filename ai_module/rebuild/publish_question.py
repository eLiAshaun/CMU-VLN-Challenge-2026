"""Publish a development question through the official String topic."""
import argparse
import json
import time

import rclpy
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question')
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('cmu_rebuild_question_publisher')
    publisher = node.create_publisher(String, '/challenge_question', 10)
    end = time.monotonic() + 10
    while publisher.get_subscription_count() == 0 and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if publisher.get_subscription_count() == 0:
        raise RuntimeError('No /challenge_question subscriber discovered')
    publisher.publish(String(data=args.question))
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    print(json.dumps({'published_question': args.question, 'topic': '/challenge_question'}))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
