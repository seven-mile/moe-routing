#!/bin/bash

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it to proceed."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    echo "On macOS (Homebrew): brew install jq"
    exit 1
fi

# Loop through all files ending in .json in the current directory
for file in $@; do
  # Check if the file exists and is a regular file
  if [ -f "$file" ]; then
    echo "--- Results for: $file ---"
    
    # Use jq to extract the required fields and format the output
    jq '{
      model_config: {
        use_assisted_topk: (.config.model_args.use_assisted_topk // false),
        assisted_action: .config.model_args.assisted_action,
        topk_assistant_model: (.config.model_args.topk_assistant_model // "self")
      },
      scores: {
        arc_challenge: {
          score: .results.arc_challenge["acc_norm,none"],
          stderr: .results.arc_challenge["acc_norm_stderr,none"]
        },
        gsm8k: {
          score_strict: .results.gsm8k["exact_match,strict-match"],
          stderr_strict: .results.gsm8k["exact_match_stderr,strict-match"],
          score_flexible: .results.gsm8k["exact_match,flexible-extract"],
          stderr_flexible: .results.gsm8k["exact_match_stderr,flexible-extract"]
        },
        gsm8k_cot: {
          score_strict: .results.gsm8k_cot["exact_match,strict-match"],
          stderr_strict: .results.gsm8k_cot["exact_match_stderr,strict-match"],
          score_flexible: .results.gsm8k_cot["exact_match,flexible-extract"],
          stderr_flexible: .results.gsm8k_cot["exact_match_stderr,flexible-extract"]
        },
        hellaswag: {
          score: .results.hellaswag["acc_norm,none"],
          stderr: .results.hellaswag["acc_norm_stderr,none"]
        },
        mmlu: {
          score: .results.mmlu["acc,none"],
          stderr: .results.mmlu["acc_stderr,none"]
        },
        cmmlu: {
          score: .results.cmmlu["acc_norm,none"],
          stderr: .results.cmmlu["acc_norm_stderr,none"]
        },
        truthfulqa: {
          score: .results.truthfulqa_mc2["acc,none"],
          stderr: .results.truthfulqa_mc2["acc_stderr,none"]
        },
        winogrande: {
          score: .results.winogrande["acc,none"],
          stderr: .results.winogrande["acc_stderr,none"]
        },
        ceval: {
          score: .results["ceval-valid"]["acc_norm,none"],
          stderr: .results["ceval-valid"]["acc_norm_stderr,none"]
        },
        humaneval: {
          score: .results.humaneval["pass@1,create_test"],
          stderr: .results.humaneval["pass@1_stderr,create_test"]
        },
        gpqa_diamond_cot_n_shot: {
          score_strict: .results.gpqa_diamond_cot_n_shot["exact_match,strict-match"],
          stderr_strict: .results.gpqa_diamond_cot_n_shot["exact_match_stderr,strict-match"],
          score_flexible: .results.gpqa_diamond_cot_n_shot["exact_match,flexible-extract"],
          stderr_flexible: .results.gpqa_diamond_cot_n_shot["exact_match_stderr,flexible-extract"]
        }
      },
      assisted_stat: {
        ksum: .assisted_stat.ksum,
        kcnt: .assisted_stat.kcnt,
        benefit: .assisted_stat.benefit
      }
    }' "$file" \
    | jq '.scores |= map_values(with_entries(select(.value != null))) | .scores |= del(.. | select(. == {}))'
    
    # Add a newline for better readability between file outputs
    echo ""
  fi
done