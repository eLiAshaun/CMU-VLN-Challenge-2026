"""High-level semantic waypoints over observed terrain and actual odometry.

The official controller still owns collision avoidance and motion. Intermediate
waypoints and trajectory checks provide empirical language-region handling.
"""
from __future__ import annotations

from collections import deque
import heapq
import math

import numpy as np

from .task_ir import evaluate, measured_extent, selection_supported


AVOID_ACTIONS = frozenset({'avoid_path', 'avoid_region'})
REGION_ACTIONS = frozenset({'near_path', 'between_path'})
OBJECT_ACTIONS = frozenset({'pass_near', 'go_to'})


def in_polygon(point, polygon) -> bool:
    x, y = point[:2]
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing = (x2-x1) * (y-y1) / (y2-y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def surface_distance(position, record) -> float:
    delta = np.abs(np.asarray(position[:2]) - record.center[:2]) - navigation_extent(record)[:2] / 2
    return float(np.linalg.norm(np.maximum(delta, 0)))


def navigation_extent(record) -> np.ndarray:
    extent = measured_extent(record)
    # An estimated location can guide acquisition/action. Its depth spread
    # cannot enlarge the physical arrival region or push approach points away.
    return np.zeros(3) if extent is None else extent


def needs_broad_coverage(expression) -> bool:
    if isinstance(expression, dict):
        op = str(expression.get('op', expression.get('operator', ''))).strip().lower().replace('-', '_')
        return op in {'argmin_distance', 'argmax_distance', 'count_distinct'} or any(
            needs_broad_coverage(value) for value in expression.values())
    return isinstance(expression, list) and any(needs_broad_coverage(value) for value in expression)


class Navigation:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.position = None
        self.stamp = 0.0
        self.step_index = 0
        self.completed_steps = []
        self.active = None
        self.route = deque()
        self.route_purpose = None
        self.observed_positions = []
        self.observation_stamp = -1.0
        self.forbidden = []
        self.violations = []
        self.total_distance = 0.0
        self.resolution = 0.25
        self.last_terrain = None
        self.free_cells = set()
        self.progress_target = None
        self.progress_distance = float('inf')
        self.progress_stamp = 0.0
        self.unproductive_routes = set()
        self.progress_position = None
        self.require_semantic_route = False
        self.avoid_focus_ids = []
        self._avoid_states = []

    def _clear_route(self) -> None:
        self.route.clear()
        self.progress_target = None
        self.progress_position = None

    def _check_route_progress(self):
        if not self.route:
            self.progress_target = None
            return None
        target = tuple(self.route[0])
        distance = float(np.linalg.norm(self.position[:2]-self.route[0]))
        translated = self.progress_position is None or np.linalg.norm(self.position[:2]-self.progress_position) >= 0.05
        if target != self.progress_target or translated:
            self.progress_target, self.progress_distance, self.progress_stamp = target, distance, self.stamp
            self.progress_position = self.position[:2].copy()
        elif self.stamp-self.progress_stamp >= self.config.get('navigation_stall_seconds', 8.0):
            origin = self._cell(self.position)
            destination = tuple(self.route[-1])
            self.unproductive_routes.add((origin, self._cell(destination)))
            # The remaining bends were planned from a leg we did not reach.
            # Replan from the current pose instead of jumping to a later bend.
            self._clear_route()
            self.progress_target = None
            return {'reason': 'waypoint_progress_stalled', 'waypoint': target,
                    'position': self.position.tolist(), 'remaining_distance_m': distance,
                    'stamp': self.stamp, 'origin_cell': origin, 'destination': destination}
        return None

    def _route_was_unproductive(self, start, goal) -> bool:
        """Reject a failed goal neighborhood from the same pose neighborhood."""
        margin = int(math.ceil(float(self.config.get('waypoint_reached_distance_m', 0.4)) /
                              self.resolution))
        for failed_start, failed_goal in self.unproductive_routes:
            if (max(abs(start[0] - failed_start[0]), abs(start[1] - failed_start[1])) <= margin
                    and max(abs(goal[0] - failed_goal[0]), abs(goal[1] - failed_goal[1])) <= margin):
                return True
        return False

    def note_observation(self, position, stamp: float) -> None:
        if stamp <= self.observation_stamp:
            return
        self.observation_stamp = stamp
        self.observed_positions.append(np.asarray(position[:2], dtype=float))
        self.observed_positions = self.observed_positions[-128:]

    def update_pose(self, position: list, stamp: float) -> None:
        point = np.asarray(position, dtype=float)
        if self.position is not None:
            self.total_distance += float(np.linalg.norm(point[:2] - self.position[:2]))
        self.position, self.stamp = point, float(stamp)
        # The official omni-directional controller stops within 0.30 m of
        # its discretized local path endpoint. Leave 0.10 m for that endpoint
        # discretization when advancing our intermediate waypoint sequence.
        reached_distance = self.config.get('waypoint_reached_distance_m', 0.4)
        while self.route and np.linalg.norm(point[:2]-self.route[0]) <= reached_distance:
            self.route.popleft()
        for state in self._avoid_states:
            inside = in_polygon(point, state['polygon'])
            if inside and not state['inside'] and state['armed']:
                if not self.violations:
                    self.violations.append({'reason': 'entered_avoid_region', 'stamp': stamp,
                                            'position': point.tolist()})
            if not inside:
                # Leaving a polygon that was occupied when its constraint was
                # bound arms the normal re-entry check without retroactively
                # treating that initial occupancy as a violation.
                state['armed'] = True
            state['inside'] = inside
        if self.active is None:
            return
        action = self.active.get('action')
        if action == 'between_path':
            self._update_between_progress(point, stamp)
            return
        record = self.active.get('record')
        if record is None or record.center is None:
            return
        distance = surface_distance(point, record)
        translation = float(np.linalg.norm(point[:2]-self.active['start_position']))
        if action == 'near_path':
            if not self.active.get('entered', False):
                if distance <= self.active['radius'] and translation >= 0.25:
                    self.active['entered'] = True
                    self.active['entry_position'] = point[:2].copy()
                    direction = point[:2] - record.center[:2]
                    if np.linalg.norm(direction) > 1e-9:
                        self.active['entry_unit'] = direction / np.linalg.norm(direction)
                    self._clear_route()
            elif distance > self.active['radius'] and translation >= 0.25:
                self._complete_object_step(record, point, stamp, distance, translation)
            return
        arrived = distance <= self.active['radius']
        if action == 'pass_near':
            arrived = arrived and translation >= 0.25
        if arrived:
            self._complete_object_step(record, point, stamp, distance, translation)

    def _complete_object_step(self, record, point, stamp: float, distance: float,
                              translation: float) -> None:
        self.completed_steps.append({'step': self.step_index, 'action': self.active['action'],
                                     'compiled_order': self.active['compiled_order'],
                                     'object_id': record.id, 'stamp': stamp,
                                     'position': point.tolist(), 'surface_distance_m': distance,
                                     'distance_basis': ('measured_extent' if measured_extent(record) is not None
                                                        else 'estimated_location'),
                                     'target_center': record.center.tolist(),
                                     'target_bbox': record.bbox.tolist(),
                                     'translation_since_binding_m': translation,
                                     'target_observation_id': (record.observation_ids[-1]
                                                               if record.observation_ids else None)})
        self.step_index += 1
        self.active = None
        self._clear_route()

    def _update_between_progress(self, point, stamp: float) -> None:
        polygon = self.active.get('polygon')
        if polygon is None:
            return
        midpoint = self.active['midpoint']
        normal = self.active['normal']
        tangent = self.active['tangent']
        relative = point[:2] - midpoint
        along = float(np.dot(relative, tangent))
        side = float(np.dot(relative, normal))
        inside = (abs(along) <= self.active['half_length'] and
                  abs(side) <= self.active['half_width'])
        if not self.active.get('entered', False):
            if inside:
                self.active['entered'] = True
                self.active['entry_position'] = point[:2].copy()
                self._clear_route()
            return
        entry_side = self.active['entry_side']
        # Require the actual trajectory to leave the opposite side of the
        # corridor. Reaching either anchor or merely touching the rectangle
        # cannot complete a between-path step.
        translation = float(np.linalg.norm(point[:2] - self.active['start_position']))
        if not inside and side * entry_side < 0 and translation >= 0.25:
            self.completed_steps.append({
                'step': self.step_index,
                'action': 'between_path',
                'compiled_order': self.active['compiled_order'],
                'anchor_ids': list(self.active['anchor_ids']),
                'stamp': stamp,
                'position': point.tolist(),
                'entry_position': self.active.get('entry_position', point[:2]).tolist(),
                'corridor_polygon': self.active['polygon'].tolist(),
                'trajectory_constraint': 'enter_through_leave',
            })
            self.step_index += 1
            self.active = None
            self._clear_route()

    def _cell(self, position) -> tuple[int, int]:
        return tuple(np.rint(np.asarray(position[:2]) / self.resolution).astype(int))

    def _point(self, cell) -> np.ndarray:
        return np.asarray(cell, dtype=float) * self.resolution

    def _terrain(self, terrain: np.ndarray) -> None:
        if terrain is self.last_terrain:
            return
        self.last_terrain = terrain
        self.free_cells = set()
        if self.position is None or not len(terrain):
            return
        points = np.asarray(terrain)
        points = points[np.isfinite(points).all(axis=1)]
        points = points[np.linalg.norm(points[:, :2]-self.position[:2], axis=1) <= 12]
        if not len(points):
            return
        height = points[:, 3] if points.shape[1] > 3 else points[:, 2]-(self.position[2]-0.75)
        cells = np.rint(points[:, :2] / self.resolution).astype(int)
        threshold = self.config.get('terrain_obstacle_height_threshold', 0.05)
        free = set(map(tuple, cells[height <= threshold]))
        obstacle = set(map(tuple, cells[height > threshold]))
        # Terrain returns are endpoints, not a dense free-space raster.
        # Ground-return rays connect the known robot footprint to observed
        # floor; occupied columns below remain authoritative after inflation.
        start = np.asarray(self._cell(self.position))
        ray_free = set()
        for cell in free:
            end = np.asarray(cell)
            count = int(np.max(np.abs(end-start))) + 1
            ray_free.update(map(tuple, np.rint(np.linspace(start, end, count)).astype(int)))
        free.update(ray_free)
        clearance = self.config.get('terrain_obstacle_clearance_m', 0.5)
        inflation = int(math.ceil(clearance/self.resolution))
        blocked = set()
        for dx in range(-inflation, inflation+1):
            for dy in range(-inflation, inflation+1):
                if math.hypot(dx, dy)*self.resolution <= clearance:
                    blocked.update((x+dx, y+dy) for x, y in obstacle)
        self.free_cells = free - blocked
        self.free_cells.add(self._cell(self.position))

    def _allowed(self, cell) -> bool:
        return cell in self.free_cells and not any(in_polygon(self._point(cell), p) for p in self.forbidden)

    def _escape_target(self) -> np.ndarray | None:
        """Find the nearest outside point when the current pose is occupied."""
        if self.position is None:
            return None
        point = self.position[:2]
        containing = [polygon for polygon in self.forbidden if in_polygon(point, polygon)]
        if not containing:
            return None
        margin = (float(self.config.get('waypoint_reached_distance_m', 0.4)) +
                  self.resolution)
        candidates = []
        for polygon in containing:
            vertices = np.asarray(polygon, dtype=float)[:, :2]
            if len(vertices) < 3:
                continue
            centroid = np.mean(vertices, axis=0)
            for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
                edge = end - start
                length_sq = float(np.dot(edge, edge))
                fraction = (float(np.dot(point-start, edge)) / length_sq
                            if length_sq > 1e-12 else 0.0)
                projection = start + np.clip(fraction, 0.0, 1.0) * edge
                outward = projection - centroid
                norm = float(np.linalg.norm(outward))
                if norm <= 1e-9:
                    continue
                target = projection + outward / norm * margin
                if any(in_polygon(target, other) for other in self.forbidden):
                    continue
                if self._route_was_unproductive(self._cell(point), self._cell(target)):
                    continue
                candidates.append((float(np.linalg.norm(target-point)), target))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _route_length(position, path) -> float:
        points = [np.asarray(position[:2], dtype=float), *[np.asarray(item[:2], dtype=float)
                                                            for item in path]]
        return float(sum(np.linalg.norm(after-before)
                         for before, after in zip(points, points[1:])))

    def _radial_target(self, center, bbox, unit, radius: float) -> np.ndarray:
        """Return a surface-relative waypoint on a selected approach ray."""
        center = np.asarray(center, dtype=float)
        extent = np.asarray(bbox, dtype=float).reshape(-1)
        unit = np.asarray(unit, dtype=float)[:2]
        norm = float(np.linalg.norm(unit))
        if norm <= 1e-9:
            unit = np.array([1.0, 0.0])
        else:
            unit = unit / norm
        footprint = 0.0
        if len(extent) >= 2 and np.isfinite(extent[:2]).all():
            footprint = float(np.min(extent[:2] / 2.0 /
                                     np.maximum(np.abs(unit), 1e-9)))
        return center[:2] + unit * (footprint + float(radius) * 0.75)

    def _approach_candidates(self, center, bbox, radius: float,
                             *, for_observation: bool = False) -> list[dict]:
        center, bbox = np.asarray(center, dtype=float), np.asarray(bbox, dtype=float)
        direction = self.position[:2] - center[:2]
        if np.linalg.norm(direction) <= 1e-9:
            direction = np.array([1.0, 0.0])
        angle = math.atan2(direction[1], direction[0])
        candidates = []
        for offset in np.linspace(-math.pi, math.pi, 17):
            unit = np.array([math.cos(angle+offset), math.sin(angle+offset)])
            target = self._radial_target(center, bbox, unit, radius)
            path = self._path(target)
            if not path:
                continue
            # Filter every viewpoint before choosing the shortest path.
            # Rejecting only the nearest result hid valid views on the other
            # sides of an anchor and stopped further observation.
            if for_observation and any(np.linalg.norm(path[-1]-p) < 0.8
                                       for p in self.observed_positions):
                continue
            candidates.append({'length': self._route_length(self.position, path),
                               'path': path, 'unit': unit, 'target': target})
        return candidates

    def _path(self, destination) -> list[np.ndarray]:
        start, goal = self._cell(self.position), self._cell(destination)
        if self._route_was_unproductive(start, goal):
            return []
        destination = np.asarray(destination, dtype=float)

        def direct() -> list[np.ndarray]:
            if any(in_polygon(destination, polygon) for polygon in self.forbidden):
                return []
            return [self._point(goal)] if goal != start else []

        escape = self._escape_target()
        if escape is not None:
            escape_cell = self._cell(escape)
            if self._route_was_unproductive(start, escape_cell):
                return []
            return [escape] if escape_cell != start else []

        # The official system owns ordinary global routing and collision
        # avoidance. An incomplete AI free-space grid must not veto its goal.
        # Explicit language regions require our constrained intermediate route.
        if not self.require_semantic_route:
            return direct()
        # Terrain returns are optional and sparse.  A semantic waypoint still
        # remains a high-level goal for FAR when no local grid is available.
        if not self.free_cells:
            return direct()
        if not self._allowed(goal):
            return []
        queue = [(math.dist(start, goal), 0.0, start)]
        costs, parents = {start: 0.0}, {}
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost > costs[cell]:
                continue
            if cell == goal:
                cells = [cell]
                while cells[-1] != start:
                    cells.append(parents[cells[-1]])
                path = [self._point(c) for c in reversed(cells)]
                # Retain bends and at most a metre between dispatch points.
                result = []
                for index in range(1, len(path)):
                    bend = index == len(path)-1 or not np.allclose(path[index]-path[index-1], path[index+1]-path[index])
                    if bend or np.linalg.norm(path[index]-(result[-1] if result else path[0])) >= 1.0:
                        result.append(path[index])
                return result
            for dx, dy in offsets:
                adjacent = (cell[0]+dx, cell[1]+dy)
                if not self._allowed(adjacent):
                    continue
                if dx and dy and (not self._allowed((cell[0]+dx, cell[1])) or not self._allowed((cell[0], cell[1]+dy))):
                    continue
                new_cost = cost + math.hypot(dx, dy)
                if new_cost < costs.get(adjacent, float('inf')):
                    costs[adjacent], parents[adjacent] = new_cost, cell
                    heapq.heappush(queue, (new_cost+math.dist(adjacent, goal), new_cost, adjacent))
        return []

    def _object_polygon(self, record, margin: float) -> np.ndarray | None:
        center = getattr(record, 'center', None)
        if center is None and isinstance(record, dict):
            center = record.get('center')
        if center is None:
            return None
        center = np.asarray(center, dtype=float).reshape(-1)
        if len(center) < 2 or not np.isfinite(center[:2]).all():
            return None
        extent = navigation_extent(record)
        extent = np.asarray(extent, dtype=float).reshape(-1)
        half = (extent[:2] / 2.0 if len(extent) >= 2 and np.isfinite(extent[:2]).all()
                else np.zeros(2, dtype=float))
        margin = max(0.0, float(margin))
        low, high = center[:2] - half - margin, center[:2] + half + margin
        return np.array([[low[0], low[1]], [high[0], low[1]],
                         [high[0], high[1]], [low[0], high[1]]], dtype=float)

    def _between_half_width(self, expression=None) -> float:
        if isinstance(expression, dict) and expression.get('corridor_width_m') is not None:
            try:
                value = float(expression['corridor_width_m'])
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and value > 0.0:
                return value
        # The query geometry uses this same bounded cross-track policy.  The
        # pass-near policy supplies the physical scale; the factor turns its
        # radial distance into a corridor half-width.
        pass_radius = float(self.config.get('pass_near_distance_m', 1.5))
        return max(0.65, min(1.5, 0.5 * pass_radius))

    def _between_geometry(self, left, right, expression=None) -> dict | None:
        left_center = getattr(left, 'center', None)
        right_center = getattr(right, 'center', None)
        if left_center is None and isinstance(left, dict):
            left_center = left.get('center')
        if right_center is None and isinstance(right, dict):
            right_center = right.get('center')
        if left_center is None or right_center is None:
            return None
        left_center = np.asarray(left_center, dtype=float).reshape(-1)
        right_center = np.asarray(right_center, dtype=float).reshape(-1)
        if len(left_center) < 2 or len(right_center) < 2:
            return None
        vector = right_center[:2] - left_center[:2]
        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length <= 1e-6:
            return None
        tangent = vector / length
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        midpoint = (left_center[:2] + right_center[:2]) / 2.0
        half_length = max(length / 2.0, self.config.get('waypoint_reached_distance_m', 0.4))
        half_width = self._between_half_width(expression)
        corners = np.array([
            midpoint - tangent * half_length - normal * half_width,
            midpoint + tangent * half_length - normal * half_width,
            midpoint + tangent * half_length + normal * half_width,
            midpoint - tangent * half_length + normal * half_width,
        ], dtype=float)
        side = float(np.dot(self.position[:2] - midpoint, normal)) if self.position is not None else 1.0
        entry_side = 1.0 if side >= 0.0 else -1.0
        return {
            'midpoint': midpoint,
            'tangent': tangent,
            'normal': normal,
            'polygon': corners,
            'half_length': float(half_length),
            'half_width': float(half_width),
            'entry_side': entry_side,
        }

    def _resolve_anchor_pair(self, anchor_expressions, records, width_expression=None):
        if not isinstance(anchor_expressions, list) or len(anchor_expressions) != 2:
            return None, [{'reason': 'between_region_requires_two_anchors'}]
        selections = [evaluate(expression, records) for expression in anchor_expressions]
        diagnostics = []
        for index, selection in enumerate(selections):
            if not selection.complete and not selection_supported(anchor_expressions[index], selection):
                diagnostics.extend(
                    [{'anchor_index': index, **item} for item in
                     (selection.missing or [{'reason': 'between_anchor_unresolved',
                                             'candidate_ids': list(selection.object_ids)}])]
                )
        if diagnostics:
            return None, diagnostics

        candidate_pairs = []
        seen_pairs = set()
        for left_id in selections[0].object_ids:
            for right_id in selections[1].object_ids:
                if left_id == right_id:
                    continue
                pair_key = tuple(sorted((str(left_id), str(right_id))))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                # Keep the expression order for the stored identity, while
                # using an unordered key because corridor geometry is not.
                candidate_pairs.append((str(left_id), str(right_id)))
        if not candidate_pairs:
            return None, [{
                'reason': 'between_anchor_pair_has_no_distinct_instances',
                'anchor_candidate_groups': [list(selection.object_ids) for selection in selections],
            }]
        if len(candidate_pairs) != 1 and (width_expression or {}).get('action') != 'between_path':
            return None, [{
                'reason': 'between_anchor_pair_ambiguous',
                'anchor_candidate_groups': [list(selection.object_ids) for selection in selections],
                'candidate_pairs': [list(pair) for pair in candidate_pairs],
            }]

        options = []
        for pair in candidate_pairs:
            anchors = [records.get(identity) for identity in pair]
            if any(anchor is None for anchor in anchors):
                continue
            geometry = self._between_geometry(anchors[0], anchors[1], width_expression)
            if geometry is None:
                continue
            path = self._path(geometry['midpoint'])
            inside = in_polygon(self.position, geometry['polygon'])
            if len(candidate_pairs) == 1 or path or inside:
                cost = 0.0 if inside else (self._route_length(self.position, path) if path
                                          else float(np.linalg.norm(self.position[:2]-geometry['midpoint'])))
                options.append((cost, pair, anchors, geometry))
        if not options:
            return None, [{'reason': 'between_geometry_or_route_unavailable'}]
        _, _, anchors, geometry = min(options, key=lambda x: (x[0], x[1]))
        return (anchors, geometry), []

    def _resolve_between_anchors(self, expression, records):
        if (not isinstance(expression, dict) or
                expression.get('op') != 'between_region'):
            return None, [{'reason': 'between_region_requires_between_region_expression'}]
        return self._resolve_anchor_pair(expression.get('anchors'), records, expression)

    def _install_forbidden(self, polygons) -> None:
        previous = list(self._avoid_states)
        states = []
        used = set()
        tolerance = float(self.config.get('waypoint_reached_distance_m', 0.4))
        for polygon in polygons:
            current_inside = (self.position is not None and
                              in_polygon(self.position, polygon))
            match = None
            for index, candidate in enumerate(previous):
                if index in used:
                    continue
                old_polygon = candidate.get('polygon')
                if (np.asarray(old_polygon).shape == np.asarray(polygon).shape and
                        np.allclose(old_polygon, polygon, atol=tolerance)):
                    match = candidate
                    used.add(index)
                    break
            if match is None or bool(match.get('inside')) != bool(current_inside):
                # A newly bound polygon, or a geometry refresh that changes
                # the current occupancy, starts from the current pose. It is
                # armed only after the robot is observed outside it.
                armed = not current_inside
            else:
                armed = bool(match.get('armed', not current_inside))
            states.append({'polygon': np.asarray(polygon, dtype=float),
                           'inside': bool(current_inside), 'armed': armed})
        self._avoid_states = states
        self.forbidden = [state['polygon'] for state in states]

    def _avoid_regions(self, task, records) -> list[dict]:
        polygons, missing, focus_ids = [], [], []
        reached_distance = self.config.get('waypoint_reached_distance_m', 0.4)
        pass_radius = self.config.get('pass_near_distance_m', 1.5)
        for step in task.get('steps', []):
            action = step.get('action')
            if action not in AVOID_ACTIONS:
                continue
            if 'polygon' in step:
                try:
                    polygon = np.asarray(step['polygon'], dtype=float)[:, :2]
                except (TypeError, ValueError, IndexError):
                    polygon = np.empty((0, 2))
                if len(polygon) >= 3 and np.isfinite(polygon).all():
                    polygons.append(polygon)
                else:
                    missing.append({'reason': 'avoid_polygon_invalid', 'action': action})
                continue
            if 'bounds' in step:
                try:
                    low, high = np.asarray(step['bounds'], dtype=float).reshape(2, -1)[:, :2]
                except (TypeError, ValueError):
                    low = high = None
                if low is not None and high is not None and np.isfinite(np.r_[low, high]).all():
                    polygons.append(np.array([[low[0], low[1]], [high[0], low[1]],
                                              [high[0], high[1]], [low[0], high[1]]]))
                else:
                    missing.append({'reason': 'avoid_bounds_invalid', 'action': action})
                continue
            expression = (step.get('target') if action == 'avoid_path'
                          else step.get('region', step.get('region_expression',
                                                           step.get('target'))))
            if not isinstance(expression, dict):
                missing.append({'reason': f'{action}_unbound'})
                continue
            if expression.get('op') == 'between_region':
                resolved, diagnostics = self._resolve_between_anchors(expression, records)
                if resolved is None:
                    missing.extend(diagnostics)
                    for anchor_expression in expression.get('anchors', []):
                        selected = evaluate(anchor_expression, records)
                        focus_ids.extend(selected.object_ids)
                    continue
                _, geometry = resolved
                # Keep the path relation's width and leave the bridge's
                # endpoint discretization as a small semantic margin.
                width = geometry['half_width'] + float(reached_distance)
                midpoint, normal = geometry['midpoint'], geometry['normal']
                half_length = geometry['half_length'] + float(reached_distance)
                tangent = np.array([-normal[1], normal[0]])
                polygons.append(np.array([
                    midpoint - tangent * half_length - normal * width,
                    midpoint + tangent * half_length - normal * width,
                    midpoint + tangent * half_length + normal * width,
                    midpoint - tangent * half_length + normal * width,
                ]))
                continue
            selected = evaluate(expression, records)
            if not selected.complete:
                missing.extend(selected.missing or [{'reason': f'{action}_unbound'}])
            focus_ids.extend(selected.object_ids)
            for object_id in selected.object_ids:
                record = records.get(object_id)
                polygon = self._object_polygon(record, pass_radius)
                if polygon is None:
                    missing.append({'reason': f'{action}_geometry_missing', 'object_id': object_id})
                else:
                    polygons.append(polygon)
        self._install_forbidden(polygons)
        self.avoid_focus_ids = list(dict.fromkeys(focus_ids))
        return missing

    def _approach(self, center, bbox, radius: float, *, for_observation: bool = False) -> list[np.ndarray]:
        candidates = self._approach_candidates(center, bbox, radius,
                                                for_observation=for_observation)
        return min(candidates, key=lambda item: item['length'])['path'] if candidates else []

    def _explore(self, result, records, proposals, focus_ids=()) -> list[np.ndarray]:
        candidates = []
        targets = []
        anchor_paths = []
        raw_ids = [*focus_ids, *result.details.get('anchor_candidates', []), *result.object_ids]
        candidate_ids = []
        pending = list(raw_ids)
        while pending:
            oid = pending.pop(0)
            if isinstance(oid, (list, tuple, set)):
                pending[0:0] = list(oid)
            elif oid is not None and oid not in candidate_ids:
                candidate_ids.append(oid)
        for oid in candidate_ids:
            record = records.get(oid, records.get(str(oid)))
            if record is not None and record.center is not None:
                targets.append(record.center[:2])
                if record.bbox is not None:
                    path = self._approach(record.center, navigation_extent(record), 1.2, for_observation=True)
                    if path:
                        cost = sum(np.linalg.norm(b-a) for a, b in zip([self.position[:2], *path], path))
                        anchor_paths.append((float(cost), path))
        if anchor_paths:
            return min(anchor_paths, key=lambda item: item[0])[1]
        # Unconfirmed regions can motivate another view, but are never bound
        # as an action target or counted as an ObjectRecord.
        proposal_paths = []
        for proposal in proposals:
            if proposal['center'] is None:
                continue
            targets.append(np.asarray(proposal['center'][:2]))
            if proposal['bbox'] is not None:
                path = self._approach(proposal['center'], navigation_extent(proposal), 1.2, for_observation=True)
                if path:
                    cost = sum(np.linalg.norm(b-a) for a, b in zip([self.position[:2], *path], path))
                    proposal_paths.append((float(cost), path))
        if proposal_paths:
            return min(proposal_paths, key=lambda item: item[0])[1]
        for need in result.missing:
            for key in ('subject_id', 'anchor_id', 'object_id'):
                record = records.get(need.get(key))
                if record is not None and record.center is not None:
                    targets.append(record.center[:2])
        for distance in (1.5, 3.0):
            for angle in np.linspace(0, 2*math.pi, 12, endpoint=False):
                point = self.position[:2]+distance*np.array([math.cos(angle), math.sin(angle)])
                # Ordinary goals are delegated to FAR, which has the global
                # map and collision handling.  Requiring a populated local
                # terrain raster here stranded unresolved instruction targets
                # even though the same point was a valid FAR goal.  Keep the
                # local gate only for an explicitly constrained route.
                if (self.require_semantic_route and self.free_cells and
                        not self._allowed(self._cell(point))):
                    continue
                novelty = min((np.linalg.norm(point-p) for p in self.observed_positions), default=distance)
                if novelty < 0.8:
                    continue
                path = self._path(point)
                if not path:
                    continue
                target_distance = min((np.linalg.norm(point-p) for p in targets), default=0)
                score = float(novelty - 0.25*distance - 0.4*target_distance)
                candidates.append((score, path))
        return max(candidates, key=lambda item: item[0])[1] if candidates else []

    @staticmethod
    def _record_id(record) -> str | None:
        identity = getattr(record, 'id', None)
        if identity is None and isinstance(record, dict):
            identity = record.get('id')
        return str(identity) if identity is not None else None

    @staticmethod
    def _canonical_id(identity, aliases) -> str | None:
        if identity is None:
            return None
        current = str(identity)
        seen = set()
        while isinstance(aliases, dict) and current in aliases and current not in seen:
            seen.add(current)
            current = str(aliases[current])
        return current

    def _refresh_active(self, records, aliases=None) -> bool:
        """Refresh geometry for a bound identity while keeping its binding."""
        if self.active is None:
            return False
        if self.active.get('action') == 'between_path':
            anchor_ids = [self._canonical_id(identity, aliases)
                          for identity in self.active.get('anchor_ids', [])]
            self.active['anchor_ids'] = anchor_ids
            anchors = [records.get(identity) for identity in anchor_ids]
            if len(anchors) != 2 or any(anchor is None or anchor.center is None
                                        for anchor in anchors):
                self.active['anchor_records'] = []
                self._clear_route()
                return False
            old_midpoint = self.active.get('midpoint')
            geometry = self._between_geometry(anchors[0], anchors[1], self.active.get('expression'))
            if geometry is None:
                self.active['anchor_records'] = anchors
                self._clear_route()
                return False
            geometry_shifted = (old_midpoint is not None and
                                np.linalg.norm(geometry['midpoint'] - old_midpoint) >=
                                self.config.get('waypoint_reached_distance_m', 0.4))
            entry_side = self.active.get('entry_side')
            self.active.update(geometry)
            if entry_side in (-1.0, 1.0):
                self.active['entry_side'] = entry_side
            self.active['anchor_records'] = anchors
            if geometry_shifted:
                self._clear_route()
            return True
        identity = self.active.get('object_id')
        if identity is None:
            identity = self._record_id(self.active.get('record'))
        identity = self._canonical_id(identity, aliases)
        self.active['object_id'] = identity
        current = records.get(identity) if identity is not None else None
        if current is None:
            self.active['record'] = None
            self._clear_route()
            return False
        previous = self.active.get('record')
        previous_center = getattr(previous, 'center', None)
        current_center = getattr(current, 'center', None)
        moved = (previous_center is not None and current_center is not None and
                 np.linalg.norm(np.asarray(current_center)[:2] - np.asarray(previous_center)[:2]) >=
                 self.config.get('waypoint_reached_distance_m', 0.4))
        self.active['record'] = current
        self.active['object_id'] = self._record_id(current) or str(identity)
        if moved:
            self._clear_route()
        return current.center is not None

    def _queue_near_path(self) -> bool:
        active = self.active
        record = active.get('record') if active is not None else None
        if record is None or record.center is None or record.bbox is None:
            return False
        radius = float(active['radius'])
        if not active.get('entered', False):
            # If binding happened while already inside the near region, the
            # action still needs to continue out of that region.  This keeps
            # near_path distinct from pass_near without inventing a target.
            if surface_distance(self.position, record) <= radius:
                direction = self.position[:2] - record.center[:2]
                if np.linalg.norm(direction) <= 1e-9:
                    direction = np.array([1.0, 0.0])
                active['entered'] = True
                active['entry_position'] = self.position[:2].copy()
                active['entry_unit'] = direction / np.linalg.norm(direction)
            else:
                candidates = self._approach_candidates(record.center, navigation_extent(record), radius)
                if not candidates:
                    return False
                candidate = min(candidates, key=lambda item: item['length'])
                active['entry_unit'] = np.asarray(candidate['unit'], dtype=float)
                active['entry_target'] = np.asarray(candidate['target'], dtype=float)
                self.route.extend(candidate['path'])
                self.route_purpose = 'near_path_entry'
                return True
        unit = np.asarray(active.get('entry_unit', [1.0, 0.0]), dtype=float)
        if np.linalg.norm(unit) <= 1e-9:
            unit = np.array([1.0, 0.0])
        unit /= np.linalg.norm(unit)
        reached_distance = float(self.config.get('waypoint_reached_distance_m', 0.4))
        extent = navigation_extent(record)
        exit_target = self._radial_target(record.center, extent, -unit,
                                          (radius + reached_distance) / 0.75)
        # _radial_target applies the existing 0.75 approach factor.  The
        # resulting surface distance is radius + waypoint_reached_distance,
        # leaving the official 0.30 m stop tolerance outside the region.
        path = self._path(exit_target)
        if not path:
            return False
        active['exit_target'] = exit_target
        self.route.extend(path)
        self.route_purpose = 'near_path_exit'
        return True

    def _between_contains(self, point) -> bool:
        if self.active is None or self.active.get('action') != 'between_path':
            return False
        relative = np.asarray(point[:2], dtype=float) - self.active['midpoint']
        return (abs(float(np.dot(relative, self.active['tangent']))) <= self.active['half_length']
                and abs(float(np.dot(relative, self.active['normal']))) <= self.active['half_width'])

    def _queue_between_path(self) -> bool:
        active = self.active
        if active is None or active.get('action') != 'between_path':
            return False
        reached_distance = float(self.config.get('waypoint_reached_distance_m', 0.4))
        if not active.get('entered', False) and self._between_contains(self.position):
            active['entered'] = True
            active['entry_position'] = self.position[:2].copy()
        if active.get('entered', False):
            cross = active['half_width'] + reached_distance
            target = active['midpoint'] - active['normal'] * active['entry_side'] * cross
            purpose = 'between_path_exit'
        else:
            # Aim into the corridor from its current side. The destination is
            # inside the rectangle, so the actual pose must cross the region
            # before the route is replaced by the opposite-side waypoint.
            # Centerline gives the bridge's 0.30 m stop and the 0.40 m
            # waypoint bookkeeping room on both sides of the corridor.
            cross = 0.0
            target = active['midpoint'] + active['normal'] * active['entry_side'] * cross
            purpose = 'between_path_entry'
        path = self._path(target)
        if not path:
            return False
        active['entry_target' if not active.get('entered', False) else 'exit_target'] = target
        self.route.extend(path)
        self.route_purpose = purpose
        return True

    def _queue_active(self) -> bool:
        if self.active is None:
            return False
        action = self.active.get('action')
        if action == 'near_path':
            return self._queue_near_path()
        if action == 'between_path':
            return self._queue_between_path()
        record = self.active.get('record')
        if record is None or record.center is None or record.bbox is None:
            return False
        path = self._approach(record.center, navigation_extent(record), self.active['radius'])
        if not path:
            return False
        self.route.extend(path)
        self.route_purpose = action
        return True

    def _activate_object(self, step, record) -> None:
        action = step['action']
        radius = (self.config.get('pass_near_distance_m', 1.5)
                  if action in {'pass_near', 'near_path'}
                  else self.config.get('go_to_distance_m', 1.25))
        self.active = {
            'record': record,
            'object_id': self._record_id(record),
            'action': action,
            'radius': float(radius),
            'start_position': self.position[:2].copy(),
            'compiled_order': step.get('order', self.step_index),
        }
        if action == 'near_path' and surface_distance(self.position, record) <= radius:
            direction = self.position[:2] - record.center[:2]
            if np.linalg.norm(direction) <= 1e-9:
                direction = np.array([1.0, 0.0])
            self.active.update(entered=True, entry_position=self.position[:2].copy(),
                               entry_unit=direction / np.linalg.norm(direction))

    def _activate_between(self, step, anchors, geometry) -> None:
        self.active = {
            'action': 'between_path',
            'anchor_ids': [self._record_id(anchor) for anchor in anchors],
            'anchor_records': list(anchors),
            'expression': step,
            'start_position': self.position[:2].copy(),
            'compiled_order': step.get('order', self.step_index),
            **geometry,
        }
        if self._between_contains(self.position):
            self.active['entered'] = True
            self.active['entry_position'] = self.position[:2].copy()

    def _bind_object_goal(self, step, selection, records) -> bool:
        """Choose a route to a member of the supported goal set atomically."""
        radius = (self.config.get('pass_near_distance_m', 1.5)
                  if step['action'] in {'pass_near', 'near_path'}
                  else self.config.get('go_to_distance_m', 1.25))
        choices = []
        for identity in selection.object_ids:
            record = records.get(identity)
            if record is None or record.center is None or record.bbox is None:
                continue
            path = self._approach(record.center, navigation_extent(record), radius)
            inside = surface_distance(self.position, record) <= radius
            if path or inside:
                cost = 0.0 if inside else self._route_length(self.position, path)
                choices.append((cost, identity, record))
        if not choices:
            return False
        _, _, record = min(choices, key=lambda x: (x[0], x[1]))
        self._activate_object(step, record)
        self._queue_active()
        return True

    def plan(self, task, records, terrain_points, remaining_seconds, query_result, proposals,
             aliases=None) -> dict:
        result = {'waypoint': None, 'complete': False, 'progress': {'step': self.step_index,
                  'completed_steps': self.completed_steps, 'travel_m': self.total_distance},
                  'evidence_need': list(query_result.missing), 'trajectory_violations': self.violations}
        if self.position is None:
            result['evidence_need'] = [{'reason': 'pose_unavailable'}]
            return result
        self._terrain(terrain_points)
        avoid_missing = self._avoid_regions(task, records)
        steps = [step for step in task.get('steps', [])
                 if step.get('action') not in AVOID_ACTIONS]
        current_action = (self.active.get('action') if self.active is not None
                          else (steps[self.step_index].get('action')
                                if self.step_index < len(steps) else None))
        self.require_semantic_route = bool(self.forbidden) or current_action in REGION_ACTIONS
        if self.active is not None:
            if not self._refresh_active(records, aliases):
                self.active = None
                self._clear_route()
            elif self.active.get('action') != 'between_path' and self.step_index < len(steps):
                expression = steps[self.step_index].get('target')
                selected = evaluate(expression, records)
                if self.active.get('object_id') not in selected.object_ids:
                    self.active = None
                    self._clear_route()
        motion_event = self._check_route_progress()
        if motion_event is not None:
            result['motion_event'] = motion_event
        if avoid_missing:
            result['evidence_need'].extend(avoid_missing)
        if self.active is not None:
            result['progress']['active_action'] = self.active.get('action')
            if self.active.get('action') == 'between_path':
                result['progress']['anchor_ids'] = list(self.active.get('anchor_ids', []))
                result['progress']['entered_region'] = bool(self.active.get('entered', False))
            else:
                result['progress']['object_id'] = self.active.get('object_id')
                result['progress']['entered_region'] = bool(self.active.get('entered', False))
        if task['task_type'] == 'instruction' and self.step_index >= len(steps):
            result['complete'] = (bool(steps) and self.active is None and
                                  not self.violations and not avoid_missing)
            return result
        occupied_forbidden = any(in_polygon(self.position, polygon) for polygon in self.forbidden)
        if (self.require_semantic_route and self.free_cells and self.route and
                not occupied_forbidden and
                any(not self._allowed(self._cell(point)) for point in list(self.route)[:2])):
            self._clear_route()
        if not self.route:
            exploration_result = query_result
            focus_ids = list(self.avoid_focus_ids) if avoid_missing else []
            if self.active is not None:
                if self.active.get('action') == 'between_path':
                    focus_ids.extend(self.active.get('anchor_ids', []))
                elif self.active.get('object_id') is not None:
                    focus_ids.append(self.active['object_id'])
            # An unresolved avoid action blocks movement until its owner is
            # observed. This preserves the constraint across target steps.
            if avoid_missing and self.active is not None:
                self._clear_route()
            if (not avoid_missing and task['task_type'] == 'instruction'
                    and self.step_index < len(steps)):
                step = steps[self.step_index]
                if self.active is not None and not self._queue_active():
                    self.active = None
                    self._clear_route()
                if self.active is None and step.get('action') == 'between_path':
                    anchor_expressions = step.get('anchors')
                    resolved, diagnostics = self._resolve_anchor_pair(anchor_expressions, records, step)
                    if resolved is not None:
                        anchors, geometry = resolved
                        self._activate_between(step, anchors, geometry)
                        self._queue_active()
                    else:
                        result['evidence_need'].extend(diagnostics)
                        for anchor_expression in anchor_expressions or []:
                            selected = evaluate(anchor_expression, records)
                            focus_ids.extend(selected.object_ids)
                elif self.active is None:
                    expression = step.get('target')
                    if isinstance(expression, dict):
                        selection = evaluate(expression, records)
                        exploration_result = selection
                        if selection_supported(expression, selection):
                            if not self._bind_object_goal(step, selection, records):
                                result['evidence_need'].append(
                                    {'reason': 'target_geometry_or_route_unavailable',
                                     'object_ids': list(selection.object_ids)})
                            focus_ids.extend(selection.object_ids)
                        else:
                            result['evidence_need'].extend(
                                selection.missing or [{'reason': 'target_ambiguous',
                                                       'object_ids': selection.object_ids}]
                            )
                            focus_ids.extend(selection.object_ids)
            if not self.route and remaining_seconds > self.config.get('answer_reserve_seconds', 70):
                self.route.extend(self._explore(exploration_result, records, proposals,
                                                focus_ids=focus_ids))
                self.route_purpose = 'observation'
        if self.route:
            point = self.route[0]
            result['waypoint'] = [float(point[0]), float(point[1]), 0.0]
            result['purpose'] = self.route_purpose
        else:
            result['exploration_exhausted'] = True
        return result
