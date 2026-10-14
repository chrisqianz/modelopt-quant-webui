"""量化参数用例。keys 用 INPUT_ORDER 里的名字，adapter 负责按位置展开。"""

QUANTIZE_CASES: dict[str, dict] = {
    "hf_id_default": {
        "model": "meta-llama/Llama-3.1-8B", "trust_remote_code": True,
        "qformat": "nvfp4", "recipe": "", "kv_cache_qformat": "nvfp4",
        "datasets": ["cnn_dailymail"], "sizes_names": ["cnn_dailymail"],
        "sizes": {"cnn_dailymail": 1024}, "custom_datasets": [],
        "calib_seq": 512, "calib_with_images": False, "export_path": "",
        "batch_size": 1, "device": "cuda", "attn_implementation": "sdpa",
        "use_fsdp2": False, "cpu_offload": False, "low_memory_mode": False,
        "gpu_max_mem_percentage": 0.0, "skip_generate": False,
        "extra_args": "",
    },
    "local_path_recipe_multi_dataset": {
        "model": "~/models/Qwen3-8B", "trust_remote_code": True,
        "qformat": "nvfp4", "recipe": "~/Model-Optimizer/modelopt_recipes/llm/nvfp4.yaml",
        "kv_cache_qformat": "int8",
        "datasets": ["cnn_dailymail", "open_platypus"],
        "sizes_names": ["cnn_dailymail", "open_platypus"],
        "sizes": {"cnn_dailymail": 512, "open_platypus": 256},
        "custom_datasets": [["mysite", 128]],
        "calib_seq": 2048, "calib_with_images": True,
        "export_path": "~/quant-webui-tasks/out/qwen3-nvfp4",
        "batch_size": 4, "device": "cuda:0", "attn_implementation": "flash_attention_2",
        "use_fsdp2": True, "cpu_offload": True, "low_memory_mode": True,
        "gpu_max_mem_percentage": 0.9, "skip_generate": True,
        "extra_args": "",
    },
    "recipe_dir_minimal": {
        "model": "hf-org/Tiny-124M", "trust_remote_code": False,
        "qformat": "nvfp4", "recipe": "~/recipes/dir/", "kv_cache_qformat": "none",
        "datasets": [], "sizes_names": [], "sizes": {}, "custom_datasets": [],
        "calib_seq": 0, "calib_with_images": False, "export_path": "",
        "batch_size": 0, "device": "cuda", "attn_implementation": "",
        "use_fsdp2": False, "cpu_offload": False, "low_memory_mode": False,
        "gpu_max_mem_percentage": 0.0, "skip_generate": False,
        "extra_args": "",
    },
    "custom_dataset_single_col": {
        "model": "hf-org/Tiny-124M", "trust_remote_code": False,
        "qformat": "nvfp4", "recipe": "", "kv_cache_qformat": "none",
        "datasets": [], "sizes_names": [], "sizes": {},
        "custom_datasets": ["tiny"],
        "calib_seq": 0, "calib_with_images": False, "export_path": "",
        "batch_size": 0, "device": "cuda", "attn_implementation": "",
        "use_fsdp2": False, "cpu_offload": False, "low_memory_mode": False,
        "gpu_max_mem_percentage": 0.0, "skip_generate": False,
        "extra_args": "",
    },
    "extra_args_unmapped": {
        "model": "hf-org/Tiny-124M", "trust_remote_code": False,
        "qformat": "nvfp4", "recipe": "", "kv_cache_qformat": "none",
        "datasets": ["cnn_dailymail"], "sizes_names": ["cnn_dailymail"],
        "sizes": {"cnn_dailymail": 256}, "custom_datasets": [],
        "calib_seq": 512, "calib_with_images": False, "export_path": "",
        "batch_size": 0, "device": "cuda", "attn_implementation": "",
        "use_fsdp2": False, "cpu_offload": False, "low_memory_mode": False,
        "gpu_max_mem_percentage": 0.0, "skip_generate": False,
        "extra_args": "--verbose --max_gpu_memory_gb 24",
    },
}

EVAL_CASES: dict[str, dict] = {
    "quant_only": {
        "quant": "~/quant-webui-tasks/out/llama-nvfp4", "baseline": "",
        "tasks": ["gsm8k", "mmlu"], "tp": 1, "port": 8001, "batch": 8,
    },
    "with_baseline_tp2": {
        "quant": "/data/exports/q1", "baseline": "/data/models/base",
        "tasks": ["gpqa_diamond_cot_zeroshot", "ifeval"], "tp": 2, "port": 8000, "batch": 1,
    },
}
