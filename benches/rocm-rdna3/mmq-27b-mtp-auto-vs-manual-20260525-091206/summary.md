# 27B MTP MMQ baseline vs auto vs manual48

- model: `Qwen3.6-27B-Q4_K_M-mtp.gguf`
- ctx: 12288
- fill/test: 4096 cached + 1024 prompt test
- chunk/ubatch: 1024
- MTP: `--spec-draft-n-max 3`, q8_0/tbq4_0 target+draft KV
- f16 prefill: enabled, stable_nkv=12288

```tsv
case	health	fill_pps	test_pps	loaded_gib	peak_gib	caps	reasons	routes
baseline	True	789.239976931687	717.6743426516525	20786679808	None	128	disabled,native,ok	{"tile:tile": 119, "vec:q8k_tbq4v_vec": 15, "vec:vec": 2}
auto	True	815.9531591889554	730.5529515338044	20784848896	None	128	auto_nonmoe_native,disabled,ok	{"tile:tile": 119, "vec:q8k_tbq4v_vec": 15, "vec:vec": 2}
manual48	True	742.8769688007997	676.0066491591507	20785115136	None	48	disabled,manual,ok	{"tile:tile": 119, "vec:q8k_tbq4v_vec": 15, "vec:vec": 2}
```

Result: baseline and auto both use native x128. Auto is essentially baseline-preserving for dense/non-MoE; manual x48 is slower.
