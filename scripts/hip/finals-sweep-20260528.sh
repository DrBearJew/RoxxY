#!/bin/bash
# Full-spectrum benchmark: PPL, prefill, decode, VRAM, KLD
# 27B ROCm + Vulkan, 35B ROCm (experimental)
set -e
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
BUILD=build-rocm-rdna3-fa
BENCH=./$BUILD/bin/llama-bench
PERPLEX=./$BUILD/bin/llama-perplexity
MODEL_27B=/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf
MODEL_35B=/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf
TEST_FILE=/tmp/test_2048.txt
OUT=benches/rocm-rdna3/finals-sweep-20260528
mkdir -p $OUT

# Base env vars for DOT4 FA
BASE_ENV="GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1"
P16_ENV="GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1"

run_ppl() {
    local label=$1 model=$2 extra_env=$3 extra_flags=$4
    echo "PPL $label..."
    timeout 60 env $BASE_ENV $P16_ENV $extra_env $PERPLEX -m $model --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off $extra_flags -f $TEST_FILE 2>/dev/null | grep '\[1\]' > $OUT/ppl_$label.txt || echo "FAILED" > $OUT/ppl_$label.txt
}

run_prefill() {
    local label=$1 model=$2 pp=$3 extra_env=$4 extra_flags=$5
    echo "Prefill $label pp$pp..."
    timeout 120 env $BASE_ENV $P16_ENV $extra_env $BENCH -m $model -ngl 99 -b 1 -ub 1 -p $pp -n 1 -o json $extra_flags 2>/dev/null > $OUT/prefill_${label}_pp${pp}.json || echo '{"error":"TIMEOUT"}' > $OUT/prefill_${label}_pp${pp}.json
}

run_decode() {
    local label=$1 model=$2 tg=$3 extra_env=$4 extra_flags=$5
    echo "Decode $label tg$tg..."
    timeout 60 env $BASE_ENV $P16_ENV $extra_env $BENCH -m $model -ngl 99 -b 1 -ub 1 -p 512 -n $tg -o json $extra_flags 2>/dev/null > $OUT/decode_${label}_tg${tg}.json || echo '{"error":"TIMEOUT"}' > $OUT/decode_${label}_tg${tg}.json
}

run_kld() {
    local label=$1 model=$2 extra_env=$3 extra_flags=$4
    local ref_label=${5:-baseline}
    echo "KLD $label vs $ref_label..."
    timeout 60 env $BASE_ENV $P16_ENV $extra_env $PERPLEX -m $model --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off $extra_flags -f $TEST_FILE 2>/dev/null | grep '\[1\]' > $OUT/kld_${label}.txt || echo "FAILED" > $OUT/kld_${label}.txt
}

# ============================
# 27B ROCm
# ============================
echo "====== 27B ROCm ======"

# Configs: baseline, p16+f16, p16+q4_0, q8_0+q4_0
declare -A CONFIGS=(
    ["baseline"]=""
    ["p16+f16"]="--extra_flags ''"
    ["p16+q4_0"]="--extra_flags --cache-type-v q4_0"
    ["q8_0+q4_0"]="--extra_flags --cache-type-k q8_0 --cache-type-v q4_0"
)

for cfg in baseline p16+f16 p16+q4_0 q8_0+q4_0; do
    extra=""
    [[ "$cfg" == "p16+q4_0" ]] && extra="--cache-type-v q4_0"
    [[ "$cfg" == "q8_0+q4_0" ]] && extra="--cache-type-k q8_0 --cache-type-v q4_0"
    
    # PPL
    run_ppl "27b_${cfg}" $MODEL_27B "" "$extra"
    
    # Prefill: pp512, pp1024, pp2048, pp128k
    for pp in 512 1024 2048; do
        run_prefill "27b_${cfg}" $MODEL_27B $pp "" "$extra"
    done
    
    # Decode: tg128, tg256
    for tg in 128 256; do
        run_decode "27b_${cfg}" $MODEL_27B $tg "" "$extra"
    done
    
    # KLD (PPL as proxy — lower is better)
    run_kld "27b_${cfg}" $MODEL_27B "" "$extra" "baseline"
done

# 128k prefill (only for winning configs)
for cfg in p16+f16 p16+q4_0; do
    extra=""
    [[ "$cfg" == "p16+q4_0" ]] && extra="--cache-type-v q4_0"
    run_prefill "27b_${cfg}" $MODEL_27B 131072 "" "$extra"
done

# ============================
# 35B ROCm (experimental)
# ============================
echo "====== 35B ROCm ======"
EXP35_ENV="LLAMA_MTP_PREFILL_CHUNK=1024 LLAMA_MTP_PREFILL_FORCE_MMQ=1 GGML_CUDA_ROCM_QUANT_PREFILL_F16=1"

for cfg in baseline p16+f16 p16+q4_0; do
    extra=""
    [[ "$cfg" == "p16+q4_0" ]] && extra="--cache-type-v q4_0"
    
    run_ppl "35b_${cfg}" $MODEL_35B "$EXP35_ENV" "$extra"
    
    for pp in 512 1024 2048; do
        run_prefill "35b_${cfg}" $MODEL_35B $pp "$EXP35_ENV" "$extra"
    done
    
    for tg in 128 256; do
        run_decode "35b_${cfg}" $MODEL_35B $tg "$EXP35_ENV" "$extra"
    done
done

for cfg in p16+f16 p16+q4_0; do
    extra=""
    [[ "$cfg" == "p16+q4_0" ]] && extra="--cache-type-v q4_0"
    run_prefill "35b_${cfg}" $MODEL_35B 131072 "$EXP35_ENV" "$extra"
done

# ============================
# VRAM snapshots
# ============================
echo "====== VRAM ======"
for cfg in baseline p16+f16 p16+q4_0 q8_0+q4_0; do
    extra=""
    [[ "$cfg" == "p16+q4_0" ]] && extra="--cache-type-v q4_0"
    [[ "$cfg" == "q8_0+q4_0" ]] && extra="--cache-type-k q8_0 --cache-type-v q4_0"
    timeout 15 env $BASE_ENV $P16_ENV $PERPLEX -m $MODEL_27B --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off $extra -f $TEST_FILE 2>&1 | grep "KV buffer size" > $OUT/vram_27b_${cfg}.txt
done

echo "DONE. Results in $OUT/"
ls -la $OUT/
