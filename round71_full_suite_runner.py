import json, os, pathlib, signal, subprocess, time
files=sorted(str(p) for p in pathlib.Path('tests').glob('test*.py'))
results=[]
status_path=pathlib.Path('ROUND71_FULL_SUITE_STATUS.json')
results_path=pathlib.Path('ROUND71_FULL_SUITE_BY_FILE_RESULTS.json')
log_path=pathlib.Path('ROUND71_FULL_SUITE_RUNNER.log')

def write_status(current=None):
    summary={s:sum(r['status']==s for r in results) for s in ['pass','fail','timeout']}
    data={**summary,'done_files':len(results),'total_files':len(files),'current':current,'updated_at':time.time()}
    status_path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    results_path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

def log(msg):
    with log_path.open('a',encoding='utf-8') as fp:
        fp.write(msg+'\n')

start=time.time(); write_status()
for f in files:
    log(f'RUN {f}'); write_status(f)
    t=time.time()
    p=subprocess.Popen(['pytest','-q',f,'--tb=short','--disable-warnings'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        out,_=p.communicate(timeout=120)
        rc=p.returncode; status='pass' if rc==0 else 'fail'
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: p.kill()
        out,_=p.communicate(); rc=124; status='timeout'
    r={'file':f,'status':status,'returncode':rc,'duration':round(time.time()-t,2),'output_tail':(out or '')[-3000:]}
    results.append(r); log(f'{status.upper()} {r["duration"]} {f}')
    if status!='pass': log((out or '')[-1200:])
    write_status()
summary={s:sum(r['status']==s for r in results) for s in ['pass','fail','timeout']}
summary.update(done_files=len(results), total_files=len(files), elapsed_sec=round(time.time()-start,2), finished=True)
path=pathlib.Path('ROUND71_FULL_SUITE_BY_FILE_SUMMARY.json')
path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
status_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
log(f'FINISHED {summary}')
