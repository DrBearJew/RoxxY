#!/usr/bin/env python3
import json, os, re, resource, signal, subprocess, threading, time, urllib.request, urllib.error
from pathlib import Path

REPO = Path('/home/mrtrent/llama.cpp-mtp-tbq4-rdna3')
ART = Path(__file__).resolve().parent
BIN = REPO / 'build-rocm/bin/llama-server'
MODEL = '/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf'
TEMPLATE = '/home/mrtrent/.pi/agent/qwen36-merged-template.jinja'
BASE_PORT = int(os.environ.get('BASE_PORT', '18680'))
# Small, staged matrix: first prove fit/decode. Long-fill prompt tests come after this passes.
CASES = [
    ('tbq4_0', 131072),
    ('tbq4_0', 204800),
    ('planar3_0', 131072),
    ('planar3_0', 204800),
    ('iso3_0', 131072),
    ('iso3_0', 204800),
]
MARKERS = {'first':'ALPHA_314159_QWEN36', 'middle':'BRAVO_271828_MTP', 'last':'CHARLIE_161803_CTX'}

def now(): return time.strftime('%Y-%m-%dT%H:%M:%S')

def vram_b():
    try:
        out = subprocess.check_output(['rocm-smi','--showmeminfo','vram'], text=True, stderr=subprocess.STDOUT, timeout=8)
        m = re.search(r'VRAM Total Used Memory \(B\):\s*(\d+)', out)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def wait_vram_below(limit=3*1024**3, timeout=180):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = vram_b()
        if last is None or last < limit:
            return {'ok': True, 'vram_b': last}
        time.sleep(2)
    return {'ok': False, 'vram_b': last}

def env_base():
    env = os.environ.copy()
    env.update({
        'HIP_VISIBLE_DEVICES': '0',
        'RDNA2_MATMUL_OPT_V1': '1',
        'GGML_CUDA_MMQ_MAX_X': '48',
        'LLAMA_MTP_PREFILL_CHUNK': '512',
        'LLAMA_MTP_PREFILL_FORCE_MMQ': '1',
        'COMPRESSED_KV_FATTN_LOG': '1',
        'LD_LIBRARY_PATH': f'{REPO}/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:' + env.get('LD_LIBRARY_PATH',''),
    })
    for k in ['TBQ4_WMMA_FATTN','COMPRESSED_KV_WMMA_FATTN','GGML_CUDA_IQ4_XS_MMQ_SCRATCH16K','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','TBQ4_COOP_SET_ROWS','TBQ4_INNERQ','TBQ4_LAYER_ADAPTIVE']:
        env.pop(k, None)
    return env

def prompt():
    lines = [f"FIRST={MARKERS['first']}"]
    for i in range(80):
        if i == 40:
            lines.append(f"MIDDLE={MARKERS['middle']}")
        lines.append(f"line {i:04d}: qwen mtp tbq4 long-context fit smoke, keep answer exact.")
    lines += [f"LAST={MARKERS['last']}", "Reply with exactly SAFE_OUTPUT and nothing else."]
    return '\n'.join(lines)

def cmd_for(port, q, ctx):
    return [str(BIN),
        '--host','127.0.0.1','--port',str(port),
        '--model', MODEL, '--ctx-size', str(ctx),
        '--flash-attn','on','--no-context-shift','--no-webui','--no-warmup',
        '--jinja','--chat-template-file',TEMPLATE,'--reasoning','off',
        '--batch-size','1024','--ubatch-size','512','--cache-ram','128',
        '--cache-type-k',q,'--cache-type-v',q,
        '--parallel','1',
        '--spec-type','draft-mtp','--spec-draft-n-max','3']

def preexec():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

def wait_health(base, proc, timeout=600):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if proc.poll() is not None:
            return {'ok': False, 'dead': True, 'rc': proc.returncode, 'last': last}
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as r:
                body = r.read().decode(errors='replace')[:240]
                if r.status == 200:
                    return {'ok': True, 'status': r.status, 'body': body}
                last = {'status': r.status, 'body': body}
        except Exception as e:
            last = {'error_type': type(e).__name__, 'error': str(e)[:240]}
        time.sleep(2)
    return {'ok': False, 'timeout': True, 'last': last}

def request(base):
    payload = {'messages':[{'role':'user','content':prompt()}], 'max_tokens':16, 'temperature':0, 'stream':False}
    samples = []
    stop = False
    def sampler():
        while not stop:
            samples.append(vram_b())
            time.sleep(1)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    try:
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=900) as r:
            raw = r.read().decode('utf-8','replace')
        obj = json.loads(raw)
        content = obj.get('choices',[{}])[0].get('message',{}).get('content')
        out = {'ok': True, 'http_status': r.status, 'wall_sec': round(time.time()-t0,3), 'usage': obj.get('usage'), 'timings': obj.get('timings'), 'content': content, 'raw_preview': raw[:1200]}
    except urllib.error.HTTPError as e:
        out = {'ok': False, 'http_status': e.code, 'wall_sec': round(time.time()-t0,3), 'error_body': e.read().decode('utf-8','replace')[:2000]}
    except Exception as e:
        out = {'ok': False, 'wall_sec': round(time.time()-t0,3), 'error_type': type(e).__name__, 'error': str(e)[:2000]}
    stop = True
    th.join(timeout=3)
    samples.append(vram_b())
    vals = [x for x in samples if x is not None]
    out['vram'] = {'samples_b': samples, 'peak_b': max(vals) if vals else None, 'after_b': samples[-1] if samples else None}
    return out

def run_case(i, q, ctx):
    name = f'{q}-ctx{ctx//1024}k'
    cdir = ART / name
    cdir.mkdir(parents=True, exist_ok=True)
    log_path = cdir / 'server.log'
    meta_path = cdir / 'meta.json'
    env = env_base()
    pre = wait_vram_below()
    port = BASE_PORT + i
    cmd = cmd_for(port, q, ctx)
    meta = {'name': name, 'quant': q, 'ctx': ctx, 'started': now(), 'cmd': cmd, 'env_subset': {k: env.get(k) for k in ['RDNA2_MATMUL_OPT_V1','GGML_CUDA_MMQ_MAX_X','LLAMA_MTP_PREFILL_CHUNK','LLAMA_MTP_PREFILL_FORCE_MMQ','COMPRESSED_KV_FATTN_LOG']}, 'pre_vram': pre}
    proc = None
    try:
        with log_path.open('wb') as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO), preexec_fn=preexec)
        base = f'http://127.0.0.1:{port}'
        t0 = time.time()
        meta['health'] = wait_health(base, proc)
        meta['startup_sec'] = round(time.time()-t0,3)
        if meta['health'].get('ok'):
            meta['request'] = request(base)
        else:
            meta['request'] = {'ok': False, 'health_failed': True, 'vram': {'peak_b': vram_b()}}
    finally:
        if proc is not None and proc.poll() is None:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(timeout=30)
        if proc is not None:
            meta['server_rc'] = proc.returncode
        meta['post_vram'] = wait_vram_below()
        text = log_path.read_text(errors='replace') if log_path.exists() else ''
        (cdir/'server.tail.log').write_text('\n'.join(text.splitlines()[-240:])+'\n')
        meta['log_counts'] = {
            'rocm_oom': text.count('ROCm error: out of memory') + text.count('out of memory'),
            'cublas_stack': text.count('ggml_cuda_op_mul_mat_cublas'),
            'kernel_vec': text.count('kernel=vec'),
            'kernel_wmma': text.count('kernel=wmma'),
            'double_free': text.count('double free') + text.count('corruption (!prev)'),
        }
        meta['selected_kernels'] = [line for line in text.splitlines() if 'ggml_cuda_fattn_log_selection' in line][-20:]
        meta['finished'] = now()
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n')
    req = meta.get('request') or {}
    timings = req.get('timings') or {}
    peak = (req.get('vram') or {}).get('peak_b') or ((meta.get('request') or {}).get('vram') or {}).get('peak_b')
    row = {
        'name': name,
        'health': meta.get('health',{}).get('ok'),
        'ok': req.get('ok'),
        'http': req.get('http_status'),
        'content': req.get('content'),
        'startup_s': meta.get('startup_sec'),
        'wall_s': req.get('wall_sec'),
        'prompt_tps': timings.get('prompt_per_second'),
        'decode_tps': timings.get('predicted_per_second'),
        'peak_gib': round(peak/1024**3, 3) if peak else None,
        'server_rc': meta.get('server_rc'),
        'oom_hits': meta.get('log_counts',{}).get('rocm_oom'),
        'dir': str(cdir),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return meta

def main():
    ART.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (q, ctx) in enumerate(CASES):
        results.append(run_case(i, q, ctx))
    (ART/'summary.json').write_text(json.dumps(results, indent=2, sort_keys=True)+'\n')
    print('ART=' + str(ART), flush=True)

if __name__ == '__main__':
    main()
