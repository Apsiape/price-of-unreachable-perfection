"""Run all finite checks sequentially. Python 3 and NumPy are required."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import subprocess, sys, json, time, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CASES=[
 ('tails','input_lamps.py'),('tails','input_tail.py'),('tails','input_profiles.py'),('tails','independent.py'),('tails','window_charge.py'),
 ('sparse','input.py'),('sparse','independent.py'),('farey','input.py'),('farey','independent.py'),
 ('caps','input_ceiling.py'),('caps','independent.py'),
 ('compression','input.py'),('compression','independent.py')]

def memory_job():
    if os.name!='nt':
        return None
    import ctypes as c
    from ctypes import wintypes as w
    class Basic(c.Structure):
        _fields_=[('PerProcessUserTimeLimit',c.c_longlong),('PerJobUserTimeLimit',c.c_longlong),('LimitFlags',w.DWORD),('MinimumWorkingSetSize',c.c_size_t),('MaximumWorkingSetSize',c.c_size_t),('ActiveProcessLimit',w.DWORD),('Affinity',c.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
    class IO(c.Structure):
        _fields_=[(n,c.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
    class Extended(c.Structure):
        _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',c.c_size_t),('JobMemoryLimit',c.c_size_t),('PeakProcessMemoryUsed',c.c_size_t),('PeakJobMemoryUsed',c.c_size_t)]
    kernel=c.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.restype=w.HANDLE
    kernel.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
    handle=kernel.CreateJobObjectW(None,None)
    info=Extended()
    info.BasicLimitInformation.LimitFlags=0x100|0x2000
    info.ProcessMemoryLimit=1400*1024*1024
    if not kernel.SetInformationJobObject(handle,9,c.byref(info),c.sizeof(info)):
        raise OSError(c.get_last_error(),'Could not set process memory limit')
    return kernel,handle

rows=[]
for topic,name in CASES:
    folder=ROOT/topic
    recorded=folder/'recorded'
    recorded.mkdir(exist_ok=True)
    log=recorded/(Path(name).stem+'.log')
    args=[sys.executable,'-u',name]
    if topic=='compression' and name=='input.py':
        args.append('input-results.json')
    started=time.monotonic()
    job=memory_job()
    with log.open('w',encoding='utf-8') as stream:
        proc=subprocess.Popen(args,cwd=folder,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        if job and not job[0].AssignProcessToJobObject(job[1],int(proc._handle)):
            proc.kill()
            raise RuntimeError('Could not apply process memory limit')
        code=proc.wait()
    if job:
        import ctypes
        job[0].CloseHandle.argtypes=[ctypes.c_void_p]
        job[0].CloseHandle(job[1])
    row={'program':topic+'/'+name,'exit_code':code,'seconds':round(time.monotonic()-started,3),'log':log.relative_to(ROOT).as_posix()}
    rows.append(row)
    print(json.dumps(row),flush=True)
    (ROOT/'run-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    if code:
        raise SystemExit(code)
print('ALL FINITE CHECK PROGRAMS COMPLETED',flush=True)
