"""One outstanding NavigateToPose action with cancellation before replacement."""
from __future__ import annotations
import math


class Nav2Transport:
    def __init__(self, node, config, client=None, goal_factory=None):
        self.node, self.config = node, config
        if client is None:
            from rclpy.action import ActionClient
            from nav2_msgs.action import NavigateToPose
            client = ActionClient(node, NavigateToPose, config['nav2_action'])
            goal_factory = NavigateToPose.Goal
        self.client, self.goal_factory = client, goal_factory
        self.desired = None
        self.target = None
        self.handle = None
        self.sending = False
        self.cancelling = False
        self.sequence = 0
        self.last_success = None
        self.completed_stamp = -1.0
        self.status = 'idle'
        self.error = None

    @property
    def busy(self):
        return self.sending or self.handle is not None

    def submit(self, target):
        target = tuple(float(value) for value in target)
        if len(target) != 3 or not all(math.isfinite(value) for value in target):
            raise ValueError('Navigation goal must be finite x, y, yaw')
        self.desired = target
        if self.handle is not None and self.target != self.desired:
            self._cancel()
        self.pump()

    def cancel(self):
        self.desired = None
        self.last_success = None
        if self.handle is not None:
            self._cancel()

    def pump(self):
        if self.busy or self.error or self.desired is None or self.desired == self.last_success:
            return
        if not self.client.server_is_ready():
            self.status = 'waiting_for_nav2'
            return
        goal = self.goal_factory()
        goal.pose.header.frame_id = self.config['world_frame']
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        x, y, yaw = self.desired
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation.z = math.sin(yaw/2)
        goal.pose.pose.orientation.w = math.cos(yaw/2)
        self.target = self.desired
        self.sequence += 1
        seq, target = self.sequence, self.target
        self.sending, self.status = True, 'sending'
        future = self.client.send_goal_async(goal)
        future.add_done_callback(lambda future: self._accepted(future, seq, target))

    def _accepted(self, future, seq, target):
        self.sending = False
        try:
            handle = future.result()
            if not handle.accepted:
                if self.desired == target:
                    self.error, self.status = 'Nav2 rejected the goal', 'rejected'
                self.pump()
                return
            self.handle, self.status = handle, 'active'
            handle.get_result_async().add_done_callback(
                lambda result: self._result(result, seq, target))
            if self.desired != target:
                self._cancel()
        except Exception as exc:
            self.error, self.status = str(exc), 'error'

    def _cancel(self):
        if self.cancelling:
            return
        self.cancelling, self.status = True, 'cancelling'
        seq = self.sequence
        self.handle.cancel_goal_async().add_done_callback(lambda future: self._cancel_response(future, seq))

    def _cancel_response(self, future, seq):
        if seq != self.sequence:
            return
        try:
            result = future.result()
            if not result.goals_canceling and self.handle is not None:
                # Do not start another goal while a cancellation was declined.
                self.error = 'Nav2 did not accept cancellation of the active goal'
        except Exception as exc:
            self.error = f'Nav2 cancellation failed: {exc}'

    def _result(self, future, seq, target):
        if seq != self.sequence:
            return
        self.handle, self.cancelling = None, False
        try:
            response = future.result()
            code = int(getattr(response.result, 'error_code', 0))
            if self.desired != target:
                self.status = 'idle'
            elif response.status == 4 and code == 0:  # GoalStatus.STATUS_SUCCEEDED
                self.last_success, self.status = target, 'succeeded'
                self.completed_stamp = self.node.get_clock().now().nanoseconds * 1e-9
            elif response.status == 5 and self.desired is None:
                self.status = 'idle'
            else:
                self.error = (f'Nav2 status={response.status}, error_code={code}: '
                              + str(getattr(response.result, 'error_msg', '')))
                self.status = 'failed'
        except Exception as exc:
            self.error, self.status = str(exc), 'error'
        self.pump()
