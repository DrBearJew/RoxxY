#!/usr/bin/env python3
import hashlib, json, os, re, resource, signal, subprocess, time, urllib.error, urllib.request
from pathlib import Path

REPO = Path('/home/mrtrent/llama.cpp-mtp-tbq4-rdna3')
ART = Path(__file__).resolve().parent
BIN = REPO / 'build-rocm/bin/llama-server'
MODEL = '/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf'
TEMPLATE = '/home/mrtrent/.pi/agent/qwen36-merged-template.jinja'
BASE_PORT = int(os.environ.get('BASE_PORT', '18920'))
SERVER_CTX = int(os.environ.get('SPARSEV_NIAH_SERVER_CTX', '32768'))
CONTEXTS = [int(x) for x in os.environ.get('SPARSEV_NIAH_CONTEXTS', '8192,16384,32768').split(',') if x]
DEPTHS = [int(x) for x in os.environ.get('SPARSEV_NIAH_DEPTHS', '10,50,90').split(',') if x]
VARIANTS = [None, 0, 1, 2, 3, 4, 5]
TAU = {None: 'off', 0: '1e-6', 1: '3e-7', 2: '1e-7', 3: '3e-8', 4: '1e-5', 5: '1e-4'}

FILLER = (
    'Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. '
    'The old notebook contained travel notes, recipes, weather observations, and ordinary descriptions of gardens, '
    'rivers, shelves, cups, doors, maps, clocks, lamps, and quiet rooms. Nothing in this filler text is important. '
    'It exists only to create a long deterministic context for retrieval testing. '
)

def token_estimate(s: str) -> int:
    return max(1, len(s) // 4)

def filler(tokens: int) -> str:
    return (FILLER * (tokens // token_estimate(FILLER) + 8))[:tokens * 5]

def target_prompt_tokens(ctx: int) -> int:
    # Leave headroom for chat template, answer, and tokenizer-estimate error.
    return max(1024, int(ctx * 0.78))

def prompt(ctx: int, depth: int, secret: str) -> str:
    total = target_prompt_tokens(ctx)
    prefix_tokens = int(total * depth / 100)
    suffix_tokens = max(1, total - prefix_tokens - 120)
    return (
        filler(prefix_tokens) +
        f'\n\nIMPORTANT NEEDLE: The secret verification code is {secret}. Remember exactly this code.\n\n' +
        filler(suffix_tokens) +
        '\n\nQuestion: What is the exact secret verification code? Answer with only the code.\n'
    )

def preexec():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

def env_for(level):
    env = os.environ.copy()
    for k in ['GGML_CUDA_SPARSE_V_DEQUANT', 'GGML_CUDA_SPARSE_V_TAU_LEVEL', 'GGML_CUDA_TBQ4_VEC_NORM_HOIST',
              'TBQ4_WMMA_FATTN', 'COMPRESSED_KV_WMMA_FATTN', 'LLAMA_MTP_PREFILL_CHUNK', 'LLAMA_MTP_PREFILL_FORCE_MMQ']:
        env.pop(k, None)
    env.update({
        'HIP_VISIBLE_DEVICES': '0',
        'RDNA2_MATMUL_OPT_V1': '1',
        'GGML_CUDA_MMQ_MAX_X': '48',
        'COMPRESSED_KV_FATTN_LOG': '1',
        'LD_LIBRARY_PATH': f'{REPO}/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:' + env.get('LD_LIBRARY_PATH', ''),
    })
    if level is not None:
        env['GGML_CUDA_SPARSE_V_DEQUANT'] = '1'
        env['GGML_CUDA_SPARSE_V_TAU_LEVEL'] = str(level)
    return env

def wait_health(port, proc, timeout=600):
    url = f'http://127.0.0.1:{port}/health'
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if proc.poll() is not None:
            return {'ok': False, 'dead': True, 'rc': proc.returncode, 'last': last}
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                body = r.read().decode('utf-8', 'replace')[:500]
                if r.status == 200:
                    return {'ok': True, 'status': r.status, 'body': body}
                last = {'status': r.status, 'body': body}
        except Exception as e:
            last = {'error_type': type(e).__name__, 'error': str(e)[:300]}
        time.sleep(2)
    return {'ok': False, 'timeout': True, 'last': last}

def request(port, text):
    payload = {'messages': [{'role': 'user', 'content': text}], 'max_tokens': 32, 'temperature': 0, 'stream': False}
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            raw = r.read().decode('utf-8', 'replace')
        obj = json.loads(raw)
        content = obj.get('choices', [{}])[0].get('message', {}).get('content', '')
        return {'ok': True, 'wall_sec': round(time.time() - t0, 3), 'content': content,
                'usage': obj.get('usage'), 'timings': obj.get('timings'), 'raw_preview': raw[:1000]}
    except urllib.error.HTTPError as e:
        return {'ok': False, 'http_status': e.code, 'wall_sec': round(time.time() - t0, 3),
                'error_body': e.read().decode('utf-8', 'replace')[:2000]}
    except Exception as e:
        return {'ok': False, 'wall_sec': round(time.time() - t0, 3), 'error_type': type(e).__name__, 'error': str(e)[:2000]}

def summarize_routes(log_path: Path):
    txt = log_path.read_text(errors='replace') if log_path.exists() else ''
    routes = {}
    for route in re.findall(r'route=([a-zA-Z0-9_]+)', txt):
        routes[route] = routes.get(route, 0) + 1
    tau_levels = sorted(set(int(x) for x in re.findall(r'sparse_v_tau_level=(\d+)', txt)))
    return {'routes': routes, 'tau_levels_logged': tau_levels, 'log_bytes': log_path.stat().st_size if log_path.exists() else 0}

def run_variant(level, port):
    name = 'sparse_off' if level is None else f'sparse_tau{level}_{TAU[level]}'
    outdir = ART / 'niah' / name
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / 'server.log'
    cmd = [str(BIN), '-m', MODEL, '--host', '127.0.0.1', '--port', str(port), '--ctx-size', str(SERVER_CTX),
           '--batch-size', '1024', '--ubatch-size', '512', '--cache-type-k', 'q8_0', '--cache-type-v', 'tbq4_0',
           '--flash-attn', 'on', '--parallel', '1', '--cache-ram', '128', '--no-context-shift', '--no-warmup',
           '--jinja', '--chat-template-file', TEMPLATE, '--reasoning', 'off']
    meta = {'name': name, 'tau': TAU[level], 'tau_level': level, 'cmd': cmd, 'tests': [], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    proc = None
    t0 = time.time()
    try:
        with log_path.open('wb') as lf:
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT, env=env_for(level), preexec_fn=preexec)
        meta['health'] = wait_health(port, proc)
        if not meta['health'].get('ok'):
            return meta
        for ctx in CONTEXTS:
            for depth in DEPTHS:
                secret = 'SPARSEV-' + hashlib.md5(f'{name}-{ctx}-{depth}'.encode()).hexdigest()[:8].upper()
                text = prompt(ctx, depth, secret)
                resp = request(port, text)
                content = resp.get('content') or ''
                passed = bool(resp.get('ok') and secret in content)
                meta['tests'].append({'ctx': ctx, 'depth': depth, 'secret': secret, 'prompt_est_tokens': token_estimate(text),
                                      'passed': passed, 'response': resp})
                print(name, 'ctx', ctx, 'depth', depth, 'PASS' if passed else 'FAIL', content[:120].replace('\n', ' '), flush=True)
    finally:
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
        if proc:
            meta['server_rc'] = proc.returncode
    meta['wall_sec'] = round(time.time() - t0, 3)
    meta['route_summary'] = summarize_routes(log_path)
    total = len(meta.get('tests', []))
    passed = sum(1 for t in meta.get('tests', []) if t.get('passed'))
    meta['passed'] = passed; meta['total'] = total
    (outdir / 'summary.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return meta

def write_summary(results):
    summary = {'server_ctx': SERVER_CTX, 'contexts': CONTEXTS, 'depths': DEPTHS, 'variants': results}
    outdir = ART / 'niah'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    lines = ['# Sparse V NIAH sweep', '', f'- server ctx: `{SERVER_CTX}`', f'- contexts: `{CONTEXTS}`', f'- depths: `{DEPTHS}`', '',
             '| variant | tau | pass | routes | tau logs |', '|---|---:|---:|---|---|']
    for r in results:
        routes = ', '.join(f'{k}:{v}' for k, v in sorted(r.get('route_summary', {}).get('routes', {}).items()))
        lines.append(f"| {r.get('name')} | {r.get('tau')} | {r.get('passed', 0)}/{r.get('total', 0)} | {routes} | {r.get('route_summary', {}).get('tau_levels_logged', [])} |")
    (outdir / 'summary.md').write_text('\n'.join(lines) + '\n')

def main():
    results = []
    for i, level in enumerate(VARIANTS):
        r = run_variant(level, BASE_PORT + i)
        results.append(r)
        write_summary(results)
    return 0 if all(r.get('passed') == r.get('total') and r.get('total', 0) > 0 for r in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
