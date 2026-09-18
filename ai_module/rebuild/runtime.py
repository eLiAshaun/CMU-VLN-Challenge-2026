"""One model worker feeding a single episode ObjectStore and executable AST."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

from .contracts import Frame, ObjectRecord, VerifiedObservation
from .geometry import make_views, lift_detection, calibrate_depth_to_scan, reprojection_evidence, sample_view_region
from .models import ModelRuntime
from .navigation import Navigation
from .object_store import ObjectStore
from .run_log import RunLog
from .task_ir import evaluate, resolve, task_concepts, task_visual_queries


def attribute_requirements(task: dict) -> dict[str, set[str]]:
    requirements: dict[str, set[str]] = {}

    def base_class(node):
        while isinstance(node, dict):
            if node.get('op') == 'filter_class':
                return node['class']
            node = node.get('source', node.get('subject', node.get('candidates')))
        return None

    def visit(value):
        if isinstance(value, dict):
            if value.get('op') == 'filter_attribute':
                concept = base_class(value)
                if concept:
                    requirements.setdefault(concept, set()).add(value['attribute'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(task['expression'])
    visit(task.get('steps', []))
    return requirements


def pair_image(view, first, second) -> np.ndarray:
    """Fit clean RGB to the VLM budget before drawing outside the object boxes."""
    image = Image.fromarray(view.image_rgb)
    boxes = (first.box_2d, second.box_2d)
    bounds = (
        max(0, int(min(box[0] for box in boxes) - 20)),
        max(0, int(min(box[1] for box in boxes) - 20)),
        min(image.width, int(max(box[2] for box in boxes) + 20)),
        min(image.height, int(max(box[3] for box in boxes) + 20)),
    )
    crop = image.crop(bounds)
    scale = 512 / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)),
                        max(1, round(crop.height * scale))), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(crop)
    for box, color in zip(boxes, ('red', 'blue')):
        draw.rectangle(((box[0]-bounds[0])*scale-2, (box[1]-bounds[1])*scale-2,
                        (box[2]-bounds[0])*scale+2, (box[3]-bounds[1])*scale+2),
                       outline=color, width=2)
    return np.asarray(crop)


def category_image(frame, view, detection, calibration, config):
    left, top, right, bottom = detection.box_2d
    margin = max(right-left, bottom-top)*0.5 + 20
    bounds = (math.floor(left-margin), max(0, math.floor(top-margin)),
              math.ceil(right+margin), min(view.image_rgb.shape[0], math.ceil(bottom+margin)))
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    scale = config['category_crop_size'] / max(width, height)
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    crop = Image.fromarray(sample_view_region(
        frame, view, calibration, bounds, size, config['panorama_vertical_fov_deg']))
    sx, sy = size[0]/width, size[1]/height
    ImageDraw.Draw(crop).rectangle(((left-bounds[0])*sx-2, (top-bounds[1])*sy-2,
                                    (right-bounds[0])*sx+2, (bottom-bounds[1])*sy+2),
                                   outline='red', width=2)
    return crop, {'source': 'same_panorama_extended_perspective', 'view_bounds': list(bounds),
                  'output_size': list(size), 'image_stamp': frame.stamp}


@dataclass(frozen=True)
class SceneState:
    """A completed observation, detached from the model worker's mutable store."""

    records: dict[str, ObjectRecord] = field(default_factory=dict)
    snapshot: dict = field(default_factory=lambda: {
        'schema_version': 'object_store_v1', 'count': 0, 'records': {}, 'aliases': {},
    })
    observation_id: str | None = None
    image_stamp: float = -1.0
    position: np.ndarray | None = None
    observations: int = 0
    proposals: list[dict] = field(default_factory=list)


@dataclass
class Episode:
    question: str
    task: dict
    log: RunLog
    store: ObjectStore
    navigation: Navigation
    started: float
    lock: threading.RLock = field(default_factory=threading.RLock)
    scene: SceneState = field(default_factory=SceneState)
    terminal: bool = False
    failure: str | None = None

    @property
    def observations(self) -> int:
        return self.scene.observations

    @property
    def last_image_stamp(self) -> float:
        return self.scene.image_stamp

    @property
    def last_observation_position(self) -> np.ndarray | None:
        return self.scene.position


class Runtime:
    def __init__(self, config: dict, *, observation_adapter=None, navigation_factory=Navigation):
        self.config = config
        self.observation_adapter = observation_adapter
        self.navigation_factory = navigation_factory
        self.models = ModelRuntime(config, log_event=self._model_event)
        self.current_log: RunLog | None = None
        self.calibration = (json.loads(Path(config['calibration']).read_text())
                            if observation_adapter is None else {})

    def _model_event(self, event: str, **fields) -> None:
        if self.current_log is not None:
            self.current_log.event(event, **fields)

    def begin(self, question: str, log: RunLog, started: float) -> Episode:
        self.current_log = log
        log.event('task_compilation_started', question=question)
        task = self.models.compile_task(question)
        log.write('task_ir.json', task)
        log.event('task_compiled', task_type=task['task_type'], concepts=task_concepts(task))
        return Episode(question, task, log, ObjectStore(self.config), self.navigation_factory(self.config), started)

    def observe(self, episode: Episode, frame: Frame) -> None:
        if episode.terminal:
            return
        self.current_log = episode.log
        observation_dir = episode.log.directory / 'observations' / frame.observation_id
        observation_dir.mkdir(parents=True, exist_ok=True)
        camera_file = 'camera_panorama.png' if self.observation_adapter is None else 'camera_rgb.png'
        Image.fromarray(frame.panorama_rgb).save(observation_dir / camera_file)
        np.save(observation_dir / 'registered_scan.npy', frame.registered_points_map)
        episode.log.write(str(observation_dir.relative_to(episode.log.directory) / 'capture.json'), {
            'observation_id': frame.observation_id, 'image_stamp': frame.stamp,
            'registered_scan_stamp': frame.scan_stamp, 'T_map_sensor': frame.T_map_sensor,
            'geometry_source': ('/registered_scan in map; /state_estimation at image time'
                                if self.observation_adapter is None else self.observation_adapter.geometry_source),
            'sensor_metadata': getattr(frame, 'metadata', {}),
        })
        episode.log.event('observation_started', observation_id=frame.observation_id,
                          image_stamp=frame.stamp, position=frame.T_map_sensor[:3, 3])
        concepts = task_concepts(episode.task)
        visual_queries = task_visual_queries(episode.task)
        views = (make_views(frame, self.calibration, self.config) if self.observation_adapter is None
                 else self.observation_adapter.make_views(frame))
        episode.log.write(str(observation_dir.relative_to(episode.log.directory) / 'views.json'), [
            {'view_id': view.view_id, 'observation_id': view.observation_id, 'stamp': view.stamp,
             'intrinsics': view.intrinsics, 'T_map_view': view.T_map_view,
             'panorama_shape': view.panorama_shape,
             'projection': ('cropped_equirectangular' if self.observation_adapter is None
                            else self.observation_adapter.projection),
             **({'panorama_vertical_fov_deg': self.config['panorama_vertical_fov_deg'],
                 'horizontal_fov_deg': self.config['horizontal_fov_deg']}
                if self.observation_adapter is None else self.observation_adapter.view_metadata(view))}
            for view in views])
        # Raw detector outputs remain proposals until their own image region
        # is verified. No historical object label can authorize a new region.
        proposals_2d = []
        detections_summary = []
        for view in views:
            if episode.terminal:
                return
            Image.fromarray(view.image_rgb).save(observation_dir / (view.view_id + '.jpg'))
            for index, det in enumerate(self.models.detect(view, concepts, visual_queries)):
                Image.fromarray(det.mask.astype(np.uint8) * 255).save(
                    observation_dir / f'{view.view_id}_mask_{index:03d}.png')
                crop, crop_metadata = (category_image(frame, view, det, self.calibration, self.config)
                                       if self.observation_adapter is None else
                                       self.observation_adapter.category_image(frame, view, det, self.config))
                crop_path = observation_dir / f'{view.view_id}_category_{index:03d}.jpg'
                crop.save(crop_path)
                entry = {'view_id': view.view_id, 'concept': det.concept,
                         'box_2d': det.box_2d, 'score': det.score, 'object_id': None,
                         'category_crop': crop_metadata}
                detections_summary.append(entry)
                proposals_2d.append((det, view, np.asarray(crop), crop_path, entry))
        batch_size = self.config['category_batch_size']
        for offset in range(0, len(proposals_2d), batch_size):
            if episode.terminal:
                return
            batch = proposals_2d[offset:offset+batch_size]
            evidence_batch = self.models.verify_categories([(crop, det.concept) for det, _, crop, _, _ in batch])
            for (det, view, _, crop_path, entry), evidence in zip(batch, evidence_batch):
                evidence.update(observation_id=frame.observation_id, stamp=frame.stamp,
                                view_id=view.view_id, source='qwen_current_instance_crop',
                                crop_path=str(crop_path.relative_to(episode.log.directory)))
                entry['category_evidence'] = evidence
                entry['memory_admission'] = evidence['verdict'] == 'yes'
                episode.log.event('category_evidence', concept=det.concept, **evidence)
        observations, observation_views, admitted_entries, uncertain = [], [], [], []
        for view in views:
            if episode.terminal:
                return
            relevant = [(det, entry) for det, candidate_view, _, _, entry in proposals_2d
                        if candidate_view is view and entry['category_evidence']['verdict'] != 'no']
            lifted = [lift_detection(det, view, frame.registered_points_map) for det, _ in relevant]
            sparse = [i for i, obs in enumerate(lifted)
                      if len(obs.measured_points) < self.config['depth_min_lidar_points']]
            if sparse and self.config.get('estimated_depth_enabled', True):
                depth_m = self.models.depth(view)
                depth_m, calibration = calibrate_depth_to_scan(depth_m, view, frame.registered_points_map)
                episode.log.event('view_depth_calibrated', observation_id=frame.observation_id,
                                  view_id=view.view_id, **calibration)
                for i in sparse:
                    lifted[i] = lift_detection(relevant[i][0], view, frame.registered_points_map,
                                               depth_m, calibration)
                del depth_m
            for obs, (det, entry) in zip(lifted, relevant):
                if self.observation_adapter is not None:
                    obs.geometry_quality['measured_sensor_source'] = self.observation_adapter.geometry_source
                    self.observation_adapter.annotate_observation(obs, det)
                obs.geometry_quality['reprojection_evidence'] = reprojection_evidence(
                    view, det.mask, episode.store.records, frame.registered_points_map)
                entry.update(center=obs.center, bbox=obs.bbox, geometry_quality=obs.geometry_quality,
                             measured_point_count=len(obs.measured_points), estimated_point_count=len(obs.estimated_points))
                evidence = entry['category_evidence']
                if evidence['verdict'] == 'yes':
                    observations.append(VerifiedObservation(**vars(obs), category_evidence=evidence))
                    observation_views.append(view)
                    admitted_entries.append(entry)
                else:
                    uncertain.append({'concept': obs.concept, 'center': None if obs.center is None else obs.center.tolist(),
                                      'bbox': None if obs.bbox is None else obs.bbox.tolist(),
                                      'geometry_quality': dict(obs.geometry_quality),
                                      'observation_id': frame.observation_id, 'view_id': view.view_id,
                                      'reason': 'category_unconfirmed', 'evidence': evidence})
        if episode.terminal:
            return
        object_ids = episode.store.update(observations)
        for entry, oid in zip(admitted_entries, object_ids):
            entry['object_id'] = oid
        attribute_calls = 0
        for concept, attributes in attribute_requirements(episode.task).items():
            candidates = evaluate({'op': 'filter_class', 'class': concept}, episode.store.records)
            for oid in candidates.object_ids:
                if episode.terminal:
                    return
                if oid not in object_ids or attribute_calls >= self.config.get('max_attribute_verifications_per_observation', 3):
                    continue
                record = episode.store.records[oid]
                missing = [key for key in attributes if record.attributes.get(key) is None]
                crops = [obs.crop_rgb for obs, identity in zip(observations, object_ids)
                         if identity == oid and obs.crop_rgb is not None and obs.crop_rgb.size]
                if not missing or not crops:
                    continue
                crop = max(crops, key=lambda image: image.shape[0]*image.shape[1])
                verified = self.models.verify_attributes(crop, concept, missing)
                if episode.terminal:
                    return
                record.attributes.update({key: value for key, value in verified.items() if value is not None})
                episode.log.event('attribute_evidence', object_id=oid, observation_id=frame.observation_id,
                                  attributes=verified, source='qwen_object_crop')
                attribute_calls += 1
        result = resolve(episode.task, episode.store.records)
        # Revisit an existing relation when a co-view resolves more source
        # pixels of its smaller object. The first low-resolution YES is not
        # permanent. Physical geometry and image evidence have separate owners.
        keys = {(need.get('subject_id'), need.get('relation'), need.get('anchor_id'))
                for need in result.missing if need.get('relation') in {'on', 'inside'}}
        for subject_id, record in episode.store.records.items():
            for prior in record.current_relation_evidence.values():
                if isinstance(prior, dict) and prior.get('relation') in {'on', 'inside'}:
                    keys.add((subject_id, prior['relation'], prior.get('anchor_id')))
        requests = []
        for subject_id, relation, anchor_id in keys:
            if subject_id not in episode.store.records or anchor_id not in episode.store.records:
                continue
            pairs = [(i, j) for i, oid in enumerate(object_ids) if oid == subject_id
                     for j, aid in enumerate(object_ids) if aid == anchor_id
                     and observations[i].view_id == observations[j].view_id]
            if not pairs:
                continue
            def quality(pair):
                return min(int(np.count_nonzero(observations[pair[0]].panorama_mask)),
                           int(np.count_nonzero(observations[pair[1]].panorama_mask)))
            i, j = max(pairs, key=quality)
            view_quality = quality((i, j))
            prior = episode.store.records[subject_id].current_relation_evidence.get(f'{relation}:{anchor_id}')
            prior_quality = prior.get('view_quality', 0) if prior else 0
            if view_quality <= prior_quality:
                continue
            pair_query = evaluate({'op': relation,
                                   'subject': {'op': 'object_ref', 'id': subject_id},
                                   'anchor': {'op': 'object_ref', 'id': anchor_id}}, episode.store.records)
            rows = pair_query.details.get('relation_evidence', [])
            if not rows or rows[0].get('evidence', {}).get('geometry_state') != 'YES':
                continue
            requests.append((prior is not None, -(view_quality-prior_quality),
                             subject_id, relation, anchor_id, i, j, view_quality, prior))
        for _, _, subject_id, relation, anchor_id, i, j, view_quality, prior in sorted(requests)[:self.config['max_pair_verifications_per_observation']]:
            if episode.terminal:
                return
            first, second = observations[i], observations[j]
            crop = pair_image(observation_views[i], first, second)
            crop_path = observation_dir / f'pair_{subject_id}_{relation}_{anchor_id}.jpg'
            Image.fromarray(crop).save(crop_path)
            evidence = self.models.verify_pair(crop, first.concept + ' in the red box',
                                               relation, second.concept + ' in the blue box')
            evidence.update(observation_id=frame.observation_id, stamp=frame.stamp,
                            view_id=first.view_id, source='qwen_same_view_pair',
                            subject_id=subject_id, anchor_id=anchor_id, relation=relation,
                            crop_path=str(crop_path.relative_to(episode.log.directory)),
                            view_quality=view_quality,
                            view_quality_metric='minimum_source_panorama_mask_pixels',
                            input_shape=list(crop.shape))
            if prior:
                evidence['history'] = (prior.get('history', []) +
                    [{key: value for key, value in prior.items() if key != 'history'}])[-3:]
                evidence['replaces_verdict'] = prior.get('verdict')
            if episode.terminal:
                return
            episode.store.records[subject_id].current_relation_evidence[f'{relation}:{anchor_id}'] = evidence
            episode.log.event('pair_evidence', **evidence)
        if episode.terminal:
            return
        # Copy bounded query geometry/metadata; representative image pixels stay
        # with the worker and on disk. No GPU work or file IO holds episode.lock.
        records = {oid: deepcopy(replace(record, representative_crops=[]))
                   for oid, record in episode.store.records.items()}
        snapshot = episode.log.store_snapshot(episode.store)
        snapshot.update(observation_id=frame.observation_id, image_stamp=frame.stamp,
                        observations=episode.observations + 1, proposals=uncertain)
        scene = SceneState(records, snapshot, frame.observation_id, frame.stamp,
                           frame.T_map_sensor[:3, 3].copy(), episode.observations + 1, uncertain)
        result = resolve(episode.task, records)
        episode.log.write(str(observation_dir.relative_to(episode.log.directory) / 'detections.json'), detections_summary)
        episode.log.write(str(observation_dir.relative_to(episode.log.directory) / 'query_result.json'), asdict(result))
        with episode.lock:
            if episode.terminal:
                return
            episode.scene = scene
        episode.log.write('object_store.json', snapshot)
        episode.log.write('query_result.json', asdict(result))
        episode.log.event('observation_consumed', observation_id=scene.observation_id,
                          image_stamp=scene.image_stamp, observations=scene.observations,
                          instances=len(observations), objects=len(records),
                          numerical_value=result.value, missing=result.missing)
