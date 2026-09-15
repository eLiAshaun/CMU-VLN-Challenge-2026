"""Host-only scoring of independently recorded live ROS outputs.

This directory is not mounted into the AI container. Ground truth is read only
after a round finishes; the existing external scorer is imported unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / 'ai_module/runs/rebuild'
REPORTS = Path(__file__).resolve().parent


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def score_case(run_id, label, scene, question_id):
    directory = RUNS / run_id
    summary = json.loads((directory/'summary.json').read_text())
    provenance = json.loads((directory/'container_provenance.json').read_text())
    events = lines(directory/'events.jsonl')
    messages = lines(RUNS/(label+'_ros_messages.jsonl'))
    actual_scene = Path(provenance['scene_mount'][0]['Source']).parent.name
    if actual_scene != scene:
        raise ValueError(f'Actual scene {actual_scene} differs from requested {scene}')
    spec = importlib.util.spec_from_file_location(
        'existing_local_evaluator', REPO.parent/'challenge_evaluator/challenge_eval.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    data = evaluator.ChallengeData(REPO/'questions')
    record = data.question(scene, question_id)
    if summary['question'] != record['question']:
        raise ValueError('Actual accepted question differs from the public question')
    start = events[0]['wall_time']
    numerical = []
    markers = []
    trajectory = []
    waypoints = []
    missing_pose_frames = 0
    for message in messages:
        seconds = message['received_time']-start
        if seconds < 0:
            continue
        topic = message['topic']
        if topic == '/numerical_response':
            numerical.append({'elapsed_seconds':seconds,'data':message['value']})
        elif topic == '/selected_object_marker':
            markers.append({'elapsed_seconds':seconds,'frame_id':message['frame'],
                            'center':message['center'],'size':message['extent'],
                            'label':message.get('label')})
        elif topic == '/state_estimation':
            if not message.get('frame_id'):
                missing_pose_frames += 1
            trajectory.append({'elapsed_seconds':seconds,'stamp':message['stamp'],
                               'frame_id':message.get('frame_id',''),
                               'position':message['position']})
        elif topic == '/way_point_with_heading':
            waypoints.append({'elapsed_seconds':seconds,'waypoint':message['waypoint']})
    if question_id in ('q4','q5') and missing_pose_frames:
        raise ValueError('Independent trajectory observer did not record coordinate frames')
    completion = summary.get('elapsed_seconds')
    if completion is None:
        completion = next(event['elapsed_seconds'] for event in reversed(events)
                          if event['event'] == 'runtime_failure')
    accepted = [event for event in events if event['event']=='question_received']
    expected_decision = {'numerical':'numerical_output',
                         'object_reference':'object_reference_output',
                         'instruction_following':'instruction_trajectory_complete'}[record['task_type']]
    capture = {
        'schema_version':'live_rebuild_probe_adapter_v1',
        'scene':scene,'question_id':question_id,'question':summary['question'],
        'task_type':record['task_type'],
        'source_run':str(directory),'source_probe':str(RUNS/(label+'_ros_messages.jsonl')),
        'source_provenance':provenance,
        'responses':{'numerical':numerical,'object_markers':markers,'waypoints':waypoints},
        'trajectory':{'samples':trajectory},
        'timing':{'terminal_seconds':completion,'elapsed_seconds':completion,
                  'time_limit_seconds':600},
        'termination':{'reason':summary['root_decision'],
                       'success':summary['root_decision'] == expected_decision and summary['failure_stage'] is None,
                       'failure_stage':summary['failure_stage']},
        'question_accepted':bool(accepted), 'question_acceptance_count':len(accepted),
        'adapter_notes':{'waypoints_not_used_as_trajectory':True,
                         'independent_pose_samples_missing_frame':missing_pose_frames,
                         'runtime_completion_does_not_establish_task_correctness':True},
    }
    score = evaluator.score_capture(capture,data)
    score.update(run=run_id,label=label,root_decision=summary['root_decision'],
                 failure_stage=summary['failure_stage'],
                 runtime_navigation=summary.get('navigation'),
                 candidate_coverage=summary.get('query_result',{}).get('details',{}).get('candidate_coverage'),
                 numerical_answer=summary.get('numerical_answer'),
                 ros_published=summary.get('ros_published'))
    output = REPORTS / label
    output.mkdir(parents=True,exist_ok=True)
    (output/'capture.json').write_text(json.dumps(capture,indent=2)+'\n')
    (output/'score.json').write_text(json.dumps(score,indent=2)+'\n')
    (output/'report.md').write_text(evaluator.render_report(score,capture))
    return score


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('label')
    parser.add_argument('scene')
    parser.add_argument('question_id')
    args=parser.parse_args()
    score=score_case(args.run,args.label,args.scene,args.question_id)
    print(json.dumps({key:score[key] for key in ('scene','question_id','run','proxy_score','max_points','details')},indent=2))
