#!/usr/bin/env python3
import json, os, re, subprocess, time
from pathlib import Path

REPO = Path('/home/mrtrent/llama.cpp-mtp-tbq4-rdna3')
ART = Path(__file__).resolve().parent
BIN = REPO / 'build-rocm/bin/llama-perplexity'
MODEL = '/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf'
DATA = REPO / 'wikitext-2-raw/wiki.test.raw'
CTX = int(os.environ.get('SPARSEV_PPL_CTX', '2048'))
CHUNKS = int(os.environ.get('SPARSEV_PPL_CHUNKS', '4'))
BASE_LOGITS = ART / f'base-off-c{CTX}-chunks{CHUNKS}.logprobs.bin'

TAU = {
    0: '1e-6',
    1: '3e-7',
    2: '1e-7',
    3: '3e-8',
    4: '1e-5',
    5: '1e-4',
}

COMMON_CMD = [
    str(BIN), '-m', MODEL, '-f', str(DATA),
    '-c', str(CTX), '--chunks', str(CHUNKS),
    '-b', '1', '-ub', '1', '-ngl', '99',
    '-ctk', 'q8_0', '-ctv', 'tbq4_0', '-fa', 'on',
]

COMMON_ENV = {
    'HIP_VISIBLE_DEVICES': '0',
    'RDNA2_MATMUL_OPT_V1': '1',
    'GGML_CUDA_MMQ_MAX_X': '48',
    'LD_LIBRARY_PATH': f'{REPO}/build-rocm/bin:/opt/rocm-7.2.3/lib:/opt/amdgpu/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', ''),
}
DROP_ENV = [
    'GGML_CUDA_SPARSE_V_DEQUANT', 'GGML_CUDA_SPARSE_V_TAU_LEVEL',
    'GGML_CUDA_TBQ4_VEC_NORM_HOIST', 'TBQ4_WMMA_FATTN', 'COMPRESSED_KV_WMMA_FATTN',
    'LLAMA_MTP_PREFILL_CHUNK', 'LLAMA_MTP_PREFILL_FORCE_MMQ', 'COMPRESSED_KV_FATTN_LOG',
]

def env_for(level):
    env = os.environ.copy()
    for k in DROP_ENV:
        env.pop(k, None)
    env.update(COMMON_ENV)
    if level is not None:
        env['GGML_CUDA_SPARSE_V_DEQUANT'] = '1'
        env['GGML_CUDA_SPARSE_V_TAU_LEVEL'] = str(level)
    return env

def run_case(name, cmd, env, log_path):
    t0 = time.time()
    meta = {'name': name, 'cmd': cmd, 'log': str(log_path), 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    with log_path.open('wb') as out:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=out, stderr=subprocess.STDOUT)
    meta['rc'] = proc.returncode
    meta['wall_sec'] = round(time.time() - t0, 3)
    text = log_path.read_text(errors='replace')
    meta.update(parse_log(text))
    return meta

def find_float(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def parse_log(text):
    out = {}
    m = re.search(r'Final estimate: PPL =\s*([0-9.]+) \+/-\s*([0-9.]+)', text)
    if m:
        out['final_ppl'] = float(m.group(1)); out['final_ppl_ci'] = float(m.group(2))
    pairs = {
        'mean_ppl_q': r'Mean PPL\(Q\)\s*:\s*([0-9.]+) ±\s*([0-9.]+)',
        'mean_ppl_base': r'Mean PPL\(base\)\s*:\s*([0-9.]+) ±\s*([0-9.]+)',
        'mean_ln_ppl_ratio': r'Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([-0-9.]+) ±\s*([0-9.]+)',
        'mean_ppl_ratio': r'Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+) ±\s*([0-9.]+)',
        'mean_ppl_diff': r'Mean PPL\(Q\)-PPL\(base\)\s*:\s*([-0-9.]+) ±\s*([0-9.]+)',
        'mean_kld': r'Mean\s+KLD:\s*([0-9.]+) ±\s*([0-9.]+)',
        'rms_delta_p_pct': r'RMS Δp\s*:\s*([0-9.]+) ±\s*([0-9.]+) %',
        'same_top_p_pct': r'Same top p:\s*([0-9.]+) ±\s*([0-9.]+) %',
    }
    for key, pat in pairs.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1)); out[key + '_ci'] = float(m.group(2))
    for key, label in [
        ('max_kld', 'Maximum KLD'), ('p999_kld', '99.9%   KLD'), ('p99_kld', '99.0%   KLD'),
        ('p95_kld', '95.0%   KLD'), ('median_kld', 'Median  KLD'), ('min_kld', 'Minimum KLD'),
        ('max_delta_p_pct', 'Maximum Δp'), ('p999_delta_p_pct', '99.9%   Δp'), ('p99_delta_p_pct', '99.0%   Δp'),
        ('median_delta_p_pct', 'Median  Δp'), ('min_delta_p_pct', 'Minimum Δp'),
    ]:
        val = find_float(re.escape(label) + r':\s*([-0-9.]+)', text)
        if val is not None:
            out[key] = val
    return out

def write_summary(results):
    summary = {'ctx': CTX, 'chunks': CHUNKS, 'cache_type_k': 'q8_0', 'cache_type_v': 'tbq4_0', 'variants': results, 'base_logits': str(BASE_LOGITS)}
    (ART / 'ppl-kld-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    lines = [
        '# Sparse V PPL/KLD sweep', '',
        f'- ctx: `{CTX}`', f'- chunks: `{CHUNKS}`',
        '- batch/ubatch: `1/1` to force decode-style VEC sparse path',
        '- cache: `q8_0` K + `tbq4_0` V, FA on', '',
        '| variant | tau | rc | wall_s | PPL(Q) | PPL(base) | PPL ratio | KLD | max KLD | RMS Δp % | same top % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in results:
        lines.append('| {name} | {tau} | {rc} | {wall:.1f} | {pplq} | {pplb} | {ratio} | {kld} | {maxkld} | {rms} | {same} |'.format(
            name=r['name'], tau=r.get('tau', 'off'), rc=r.get('rc'), wall=r.get('wall_sec', 0.0),
            pplq=fmt(r.get('mean_ppl_q', r.get('final_ppl'))), pplb=fmt(r.get('mean_ppl_base')),
            ratio=fmt(r.get('mean_ppl_ratio')), kld=fmt(r.get('mean_kld')), maxkld=fmt(r.get('max_kld')),
            rms=fmt(r.get('rms_delta_p_pct')), same=fmt(r.get('same_top_p_pct'))))
    (ART / 'ppl-kld-summary.md').write_text('\n'.join(lines) + '\n')

def fmt(v):
    return 'n/a' if v is None else f'{v:.6g}'

def main():
    ART.mkdir(parents=True, exist_ok=True)
    results = []
    base_cmd = COMMON_CMD + ['--save-all-logits', str(BASE_LOGITS)]
    base = run_case('sparse_off_base_save', base_cmd, env_for(None), ART / 'ppl-kld-sparse-off-save.log')
    base['sparse'] = False; base['tau'] = 'off'; base['logprob_bytes'] = BASE_LOGITS.stat().st_size if BASE_LOGITS.exists() else 0
    results.append(base)
    write_summary(results)
    if base['rc'] != 0:
        return base['rc']
    for level, tau in TAU.items():
        name = f'sparse_tau{level}_{tau}'
        cmd = COMMON_CMD + ['--kl-divergence', '--kl-divergence-base', str(BASE_LOGITS)]
        r = run_case(name, cmd, env_for(level), ART / f'ppl-kld-tau{level}.log')
        r['sparse'] = True; r['tau_level'] = level; r['tau'] = tau
        results.append(r)
        write_summary(results)
    return 0 if all(r.get('rc') == 0 for r in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
