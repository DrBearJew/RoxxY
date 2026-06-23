# 27B MMQ auto smoke

Model: Qwen3.6-27B-Q4_K_M.gguf
Shape: p2048 n64 b1024 ub1024 q8_0/tbq4_0 FA on
Build: build-rocm-rdna2-fa

Results:

case	status	prompt_ts	gen_ts	selected_caps	reasons
auto	0	834.388361	26.933828	128	auto_nonmoe_native
manual48	0	778.217797	26.825279	48	manual

Conclusion:  selected native x128 for dense/non-MoE and beat manual x48 on prefill in this smoke (~834 vs ~778 tok/s). Decode effectively unchanged (~26.9 tok/s).
