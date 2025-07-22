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
        assistant_ppl_to_k: .config.model_args.assistant_ppl_to_k,
        random_shuffle_topk: (.config.model_args.random_shuffle_topk // false),
        assisted_topk_mask_layer_range: .config.model_args.assisted_topk_mask_layer_range
      },
      scores: {
        hellaswag: {
          score: .results.hellaswag["acc_norm,none"],
          stderr: .results.hellaswag["acc_norm_stderr,none"]
        },
        mmlu: {
          score: .results.mmlu["acc,none"],
          stderr: .results.mmlu["acc_stderr,none"]
        },
        winogrande: {
          score: .results.winogrande["acc,none"],
          stderr: .results.winogrande["acc_stderr,none"]
        },
        ceval: {
          score: .results["ceval-valid"]["acc_norm,none"],
          stderr: .results["ceval-valid"]["acc_norm_stderr,none"]
        }
      }
    }' "$file"
    
    # Add a newline for better readability between file outputs
    echo ""
  fi
done