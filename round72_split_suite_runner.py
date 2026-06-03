import json, pathlib, subprocess, time, math, os, signal
files=sorted(str(p) for p in pathlib.Path('tests').glob('test*.py'))
chunk_size=12
chunks=[files[i:i+chunk_size] for i in range(0,len(files),chunk_size)]
results=[]
status_path=pathlib.Path('ROUND72_SPLIT_SUITE_STATUS.json')
results_path=pathlib.Path('ROUND72_SPLIT_SUITE_RESULTS.json')
log_path=pathlib.Path('ROUND72_SPLIT_SUITE_RUNNER.log')

def write_status(current=None):
    summary={s:sum(r['status']==s for r in results) for s in ['pass','fail','timeout']}
    data={**summary,'done_chunks':len(results),'total_chunks':len(chunks),'total_files':len(files),'current':current,'updated_at':time.time()}
    status_path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    results_path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

def log(msg):
    with log_path.open('a',encoding='utf-8') as fp:
        fp.write(msg+'\n')

start=time.time(); write_status()
for idx, chunk in enumerate(chunks, 1):
    current={'chunk_index': idx, 'file_count': len(chunk), 'first_file': chunk[0], 'last_file': chunk[-1]}
    log(f'RUN chunk={idx}/{len(chunks)} files={len(chunk)} first={chunk[0]} last={chunk[-1]}')
    write_status(current)
    t=time.time()
    p=subprocess.Popen(['pytest','-q',*chunk,'--tb=short','--disable-warnings'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        out,_=p.communicate(timeout=240)
        rc=p.returncode; status='pass' if rc==0 else 'fail'
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: p.kill()
        out,_=p.communicate(); rc=124; status='timeout'
    result={'chunk_index':idx,'status':status,'returncode':rc,'duration':round(time.time()-t,2),'files':chunk,'output_tail':(out or '')[-4000:]}
    results.append(result)
    log(f'{status.upper()} duration={result["duration"]} chunk={idx}')
    if status!='pass': log((out or '')[-1600:])
    write_status()
summary={s:sum(r['status']==s for r in results) for s in ['pass','fail','timeout']}
summary.update(done_chunks=len(results), total_chunks=len(chunks), total_files=len(files), elapsed_sec=round(time.time()-start,2), finished=True)
path=pathlib.Path('ROUND72_SPLIT_SUITE_SUMMARY.json')
path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
status_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
log(f'FINISHED {summary}')
print(json.dumps(summary, ensure_ascii=False, indent=2))
