#!/usr/bin/env python3
import json, os, re, resource, signal, subprocess, time, urllib.request, urllib.error, hashlib
from pathlib import Path

REPO = Path('/home/mrtrent/llama.cpp-mtp-tbq4-rdna3')
ART = Path(__file__).resolve().parent
BIN = REPO / 'build-rocm/bin/llama-server'
MODEL = '/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf'
TEMPLATE = '/home/mrtrent/.pi/agent/qwen36-merged-template.jinja'
BASE_PORT = int(os.environ.get('BASE_PORT', '18820'))

FILLER = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. The old notebook contained travel notes, recipes, weather observations, and ordinary descriptions of gardens, rivers, shelves, cups, doors, maps, clocks, lamps, and quiet rooms. Nothing in this filler text is important. It exists only to create a long deterministic context for retrieval testing. """

def token_estimate(s: str) -> int:
    return max(1, len(s)//4)

def filler(tokens: int) -> str:
    return (FILLER * (tokens // token_estimate(FILLER) + 4))[:tokens*5]

def prompt(ctx_tokens: int, depth: int, secret: str) -> str:
    prefix_tokens = int(ctx_tokens * depth / 100)
    suffix_tokens = max(1, ctx_tokens - prefix_tokens - 80)
    return (
        filler(prefix_tokens) +
        f"\n\nIMPORTANT NEEDLE: The secret verification code is {secret}. Remember exactly this code.\n\n" +
        filler(suffix_tokens) +
        "\n\nQuestion: What is the exact secret verification code? Answer with only the code.\n"
    )

def preexec():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))

def wait_health(port, proc, timeout=420):
    url=f'http://127.0.0.1:{port}/health'
    deadline=time.time()+timeout
    last=None
    while time.time()<deadline:
        if proc.poll() is not None:
            return {'ok':False,'dead':True,'rc':proc.returncode,'last':last}
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                body=r.read().decode('utf-8','replace')[:500]
                if r.status == 200:
                    return {'ok':True,'status':r.status,'body':body}
                last={'status':r.status,'body':body}
        except Exception as e:
            last={'error_type':type(e).__name__,'error':str(e)[:300]}
        time.sleep(2)
    return {'ok':False,'timeout':True,'last':last}

def request(port, text):
    payload={'messages':[{'role':'user','content':text}], 'max_tokens':32, 'temperature':0, 'stream':False}
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    t0=time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw=r.read().decode('utf-8','replace')
        obj=json.loads(raw)
        content=obj.get('choices',[{}])[0].get('message',{}).get('content','')
        return {'ok':True,'wall_sec':round(time.time()-t0,3),'content':content,'usage':obj.get('usage'),'timings':obj.get('timings'),'raw_preview':raw[:1000]}
    except urllib.error.HTTPError as e:
        return {'ok':False,'http_status':e.code,'wall_sec':round(time.time()-t0,3),'error_body':e.read().decode('utf-8','replace')[:2000]}
    except Exception as e:
        return {'ok':False,'wall_sec':round(time.time()-t0,3),'error_type':type(e).__name__,'error':str(e)[:2000]}

def env_for(sparse: bool):
    env=os.environ.copy()
    env.update({
        'HIP_VISIBLE_DEVICES':'0',
        'RDNA2_MATMUL_OPT_V1':'1',
        'GGML_CUDA_MMQ_MAX_X':'48',
        'COMPRESSED_KV_FATTN_LOG':'1',
        'LD_LIBRARY_PATH':f'{REPO}/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:' + env.get('LD_LIBRARY_PATH',''),
    })
    if sparse:
        env['GGML_CUDA_SPARSE_V_DEQUANT']='1'
    else:
        env.pop('GGML_CUDA_SPARSE_V_DEQUANT', None)
    for k in ['GGML_CUDA_TBQ4_VEC_NORM_HOIST','TBQ4_WMMA_FATTN','COMPRESSED_KV_WMMA_FATTN','LLAMA_MTP_PREFILL_CHUNK','LLAMA_MTP_PREFILL_FORCE_MMQ']:
        env.pop(k, None)
    return env

def run_variant(name, sparse, port):
    outdir=ART/'niah'/name
    outdir.mkdir(parents=True, exist_ok=True)
    log_path=outdir/'server.log'
    cmd=[str(BIN), '-m', MODEL, '--host','127.0.0.1','--port',str(port), '--ctx-size','8192', '--batch-size','1024', '--ubatch-size','512', '--cache-type-k','q8_0','--cache-type-v','tbq4_0','--flash-attn','on','--parallel','1','--cache-ram','128','--no-context-shift','--no-warmup','--jinja','--chat-template-file',TEMPLATE,'--reasoning','off']
    meta={'name':name,'sparse':sparse,'cmd':cmd,'tests':[]}
    proc=None
    try:
        with log_path.open('wb') as lf:
            proc=subprocess.Popen(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT, env=env_for(sparse), preexec_fn=preexec)
        meta['health']=wait_health(port, proc)
        if not meta['health'].get('ok'):
            return meta
        for depth in [10,50,90]:
            secret='SPARSEV-' + hashlib.md5(f'8192-{depth}'.encode()).hexdigest()[:8].upper()
            text=prompt(6200, depth, secret)
            resp=request(port, text)
            passed=resp.get('ok') and secret in (resp.get('content') or '')
            meta['tests'].append({'ctx':8192,'depth':depth,'secret':secret,'prompt_est_tokens':token_estimate(text),'passed':bool(passed),'response':resp})
            print(name, depth, 'PASS' if passed else 'FAIL', (resp.get('content') or '')[:160].replace('\n',' '), flush=True)
    finally:
        if proc and proc.poll() is None:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(timeout=30)
        if proc:
            meta['server_rc']=proc.returncode
    (outdir/'summary.json').write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n')
    return meta

def main():
    results=[run_variant('q8k_tbq4v_sparse_off', False, BASE_PORT), run_variant('q8k_tbq4v_sparse_on', True, BASE_PORT+1)]
    summary={'results':results}
    (ART/'niah'/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    for r in results:
        passed=sum(1 for t in r.get('tests',[]) if t.get('passed'))
        total=len(r.get('tests',[]))
        print(f"{r['name']}: {passed}/{total}")
    return 0 if all(all(t.get('passed') for t in r.get('tests',[])) for r in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
