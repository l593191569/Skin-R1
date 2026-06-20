# Print and persist RUN_DIR for downstream data_construction scripts.
# Usage: source scripts/_run_dir_hint.sh && write_run_dir_hint "/path/to/run"

write_run_dir_hint() {
  local run_dir="$1"
  local outputs_dir
  outputs_dir="$(dirname "$run_dir")"

  mkdir -p "$outputs_dir"
  cat > "$outputs_dir/.env_run" <<EOF
# Source this file after a PDF / continue run to set RUN_DIR for later stages:
#   source data/outputs/.env_run
export RUN_DIR="${run_dir}"
EOF

  echo ""
  echo "================================================================"
  echo "  Next steps — set RUN_DIR for all later Stage 0 scripts:"
  echo ""
  echo "    export RUN_DIR=\"${run_dir}\""
  echo "    # or from repo root:"
  echo "    source ${outputs_dir}/.env_run"
  echo "================================================================"
  echo ""
}
