"""Run the preselected fixed-image matrix, retaining failed rounds as evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from score_live_case import score_case, REPO, REPORTS, RUNS


parser=argparse.ArgumentParser()
parser.add_argument('--after-pid',type=int)
parser.add_argument('--manifest',type=Path,default=REPORTS/'manifest.json')
parser.add_argument('--progress',type=Path,default=REPORTS/'progress.json')
args=parser.parse_args()
if args.after_pid:
    while True:
        try:
            os.kill(args.after_pid,0)
        except ProcessLookupError:
            break
        time.sleep(3)
manifest=json.loads(args.manifest.read_text())
results=[]
for index,case in enumerate(manifest['cases']):
    print(json.dumps({'event':'matrix_case_started','index':index+1,**case}),flush=True)
    command=[sys.executable,str(RUNS/'live_validation_case.py'),case['scene'],case['label'],
             case['question'],'--image',manifest['image'],'--build-log',manifest['build_log']]
    result=dict(case)
    with (REPORTS/(case['label']+'_driver.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                event=json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') in ('case_started','case_finished'):
                result.update(event)
        result['driver_exit_code']=process.wait()
    if result.get('run') and (RUNS/result['run']/'summary.json').exists():
        try:
            score=score_case(result['run'],case['label'],case['scene'],case['question_id'])
            result.update(proxy_score=score['proxy_score'],max_points=score['max_points'],
                          metrics=score['details'],failure_stage=score['failure_stage'],
                          root_decision=score['root_decision'])
        except Exception as exc:
            result['scoring_error']=repr(exc)
    else:
        result['status']='driver_or_runtime_failed_without_summary'
    results.append(result)
    temporary=args.progress.with_suffix('.tmp')
    temporary.write_text(json.dumps(results,indent=2)+'\n')
    temporary.replace(args.progress)
    print(json.dumps({'event':'matrix_case_assessed',**result}),flush=True)
print(json.dumps({'event':'matrix_finished','cases':len(results)}),flush=True)
