import json, pathlib, subprocess, time, os, signal
files=sorted(str(p) for p in pathlib.Path('tests').glob('test*.py'))
chunk_size=10
chunks=[files[i:i+chunk_size] for i in range(0,len(files),chunk_size)]
results=[]
status_path=pathlib.Path('ROUND77_SPLIT_SUITE_STATUS.json')
results_path=pathlib.Path('ROUND77_SPLIT_SUITE_BY_CHUNK_RESULTS.json')
chunks_path=pathlib.Path('ROUND77_SPLIT_CHUNKS.json')
log_path=pathlib.Path('ROUND77_SPLIT_SUITE_RUNNER.log')
chunks_path.write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding='utf-8')

def summarize(finished=False,current=None):
    passed=sum(1 for r in results if r['status']=='pass')
    failed=sum(1 for r in results if r['status']=='fail')
    timed=sum(1 for r in results if r['status']=='timeout')
    tests=sum(int(r.get('passed_tests',0) or 0) for r in results)
    return {'pass':passed,'fail':failed,'timeout':timed,'done_chunks':len(results),'total_chunks':len(chunks),'total_files':len(files),'passed_tests_by_chunk_sum':tests,'finished':finished,'current':current,'updated_at':time.time()}

def write(finished=False,current=None):
    status_path.write_text(json.dumps(summarize(finished,current),ensure_ascii=False,indent=2),encoding='utf-8')
    results_path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

def log(msg):
    with log_path.open('a',encoding='utf-8') as fp: fp.write(msg+'\n')

def parse_passed(out):
    import re
    m=re.search(r'(\d+) passed', out or '')
    return int(m.group(1)) if m else 0

start=time.time(); write()
for idx, chunk in enumerate(chunks,1):
    current={'chunk_index':idx,'file_count':len(chunk),'first_file':chunk[0],'last_file':chunk[-1]}
    log(f'RUN chunk={idx}/{len(chunks)} files={len(chunk)} first={chunk[0]} last={chunk[-1]}')
    write(False,current)
    t=time.time()
    p=subprocess.Popen(['pytest','-q',*chunk,'--tb=short','--disable-warnings'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        out,_=p.communicate(timeout=240)
        rc=p.returncode; status='pass' if rc==0 else 'fail'
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: p.kill()
        out,_=p.communicate(); rc=124; status='timeout'
    result={'chunk_index':idx,'status':status,'returncode':rc,'duration':round(time.time()-t,2),'passed_tests':parse_passed(out),'files':chunk,'output_tail':(out or '')[-5000:]}
    results.append(result)
    log(f'{status.upper()} duration={result["duration"]} passed_tests={result["passed_tests"]} chunk={idx}')
    if status!='pass': log((out or '')[-2000:])
    write()
    if status!='pass': break
summary=summarize(finished=(len(results)==len(chunks) and all(r['status']=='pass' for r in results)))
summary['elapsed_sec']=round(time.time()-start,2)
status_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
results_path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
log(f'FINISHED {summary}')
print(json.dumps(summary,ensure_ascii=False,indent=2))
