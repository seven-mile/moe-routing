HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PROMPT="${PROMPT:-如何评价《六国论》？}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

ACTION_FUNCTION="spec_with_list_layer_range"
if [[ -n "${CONSTANT_K:-}" ]]; then
  ACTION_FUNCTION="constant_k"
  ACTION_ARGS="[${CONSTANT_K}]"
elif [[ -n "${NAIVE_LOWER_LAYERS:-}" && -n "${NAIVE_K:-}" ]]; then
  BASE_K="${BASE_K:-8}"
  vals=()
  for ((i = NAIVE_K; i < BASE_K; i++)); do vals+=("1.0e30"); done
  for ((i = 0; i < NAIVE_K; i++)); do vals+=("0.0"); done
  NAIVE_CFG="$(IFS=,; echo "${vals[*]}")"
  ACTION_ARGS="[[${NAIVE_CFG}], [${NAIVE_LOWER_LAYERS}, 1000]]"
else
  ACTION_ARGS="[[16.22663120720821, 16.22663120720821, 11.861973917295447, 11.861973917295447, 7.394487721217839], [0, 0]]"
fi

REQUEST_BODY="$(cat <<EOF
{
  "model": "${MODEL}",
  "prompt": "${PROMPT}",
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "temperature": 0.0,
  "do_sample": false,
  "dyn_assisted_action_config": {
    "file": "configs/ppl_to_ks.py",
    "function": "${ACTION_FUNCTION}",
    "args": ${ACTION_ARGS}
  },
  "return_token_ids": true
}
EOF
)"

curl -X POST "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -N \
  --data "${REQUEST_BODY}"
