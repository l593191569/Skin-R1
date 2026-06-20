# Shared SKIN_R1_DATA_ROOT resolution for model_training shell scripts.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/_data_root.sh" && skin_r1_set_data_root "$PROJECT_DIR"

skin_r1_set_data_root() {
  local project_dir="$1"
  local default_root="${project_dir}/data"
  local legacy_root="${project_dir}/dataset"
  local root="${SKIN_R1_DATA_ROOT:-$default_root}"

  # Migrate stale exports that still point at the old model_training/dataset tree.
  case "${root%/}" in
    "$legacy_root"|"${legacy_root%/}")
      root="$default_root"
      ;;
  esac

  export SKIN_R1_DATA_ROOT="$root"
}
