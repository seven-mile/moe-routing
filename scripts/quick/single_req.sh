curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -N \
  --data '{
    "model": "Qwen/Qwen3-30B-A3B",
    "prompt": "如何评价《六国论》？",
                "max_tokens": 32,
                "temperature": 0.0,
                "do_sample": false,
                "dyn_assisted_action_config": {
                    "file": "configs/ppl_to_ks.py",
                    "function": "spec_with_list_layer_range",
                    "args": [
                        [10.0, 6.58, 1.275, 1.0],
                        [0, 0]
                    ]
                },
                "return_token_ids": true
  }'
