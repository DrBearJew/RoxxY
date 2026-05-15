#!/usr/bin/env python3
"""Shape metadata keys for future Triton autotune experiments.

No autotuner dependency is imported here. This file records the metadata contract
borrowed from IBM/vLLM patterns plus compressed-KV-specific keys.
"""

from __future__ import annotations

import json


AUTOTUNE_METADATA_KEYS = [
    "MAX_SEQ_Q",
    "MAX_SEQ_K",
    "AVG_SEQ_Q",
    "AVG_SEQ_K",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_D",
    "HEAD_SIZE_PADDED",
    "BLOCK_SIZE",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "Q_PER_KV",
    "NUM_SEGMENTS_PER_SEQ",
    "SLIDING_WINDOW",
    "CAUSAL",
    "FORMAT",
    "FORMAT_DOMAIN",
    "D_HEAD",
    "GFX_TARGET",
]

FIXED_RDNA3_CONFIGS = {
    "paged_materializer": {"BLOCK_D": "next_power_of_2(D_HEAD)", "BLOCK_SIZE": 4},
    "qk_2d_tiled": {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_SIZE": 4},
    "qkv_2d_tiled": {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_SIZE": 4},
    "varlen_qkv": {"BLOCK_SIZE": 4, "NUM_Q_HEADS": 4, "NUM_KV_HEADS": 2},
    "segmented_qkv": {"BLOCK_SIZE": 4, "NUM_SEGMENTS_PER_SEQ": 4},
}

FORMAT_DOMAINS = {
    "planar3_0": "original",
    "iso3_0": "original",
    "tbq4_0": "fwht_prototype_domain_requires_production_q_o_rotation",
}


def metadata_report() -> dict[str, object]:
    return {
        "result": "PASS",
        "autotuner_dependency": "none",
        "keys": AUTOTUNE_METADATA_KEYS,
        "fixed_rdna3_configs": FIXED_RDNA3_CONFIGS,
        "format_domains": FORMAT_DOMAINS,
    }


def main() -> int:
    print(json.dumps(metadata_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
