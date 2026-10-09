#!/bin/bash
# setup_daos_models.sh - Create and configure DAOS container for HuggingFace models
#
# Usage:
#   ./scripts/setup_daos_models.sh [command] [args]
#
# Commands:
#   status              Show DAOS pool and container status
#   create              Create the DAOS container (one-time setup)
#   mount               Mount container to local path (UAN only)
#   unmount             Unmount container
#   list                List available models in local hub and DAOS
#   copy <model_id>     Copy a specific model (e.g., 'allenai/OLMo-1B-0724-hf')
#   copy-all            Copy all models found in source hub
#   help                Show this help message
#
# Prerequisites:
#   - DAOS pool "AuroraGPT" must be allocated
#   - Run from login node (UAN) or compute node with daos_user_fs
#   - module load daos

set -e

# Configuration
DAOS_POOL="${DAOS_POOL:-AuroraGPT}"
DAOS_CONT="${DAOS_CONT:-prism_models}"
MOUNT_BASE="/tmp/${USER}/${DAOS_POOL}"
MOUNT_PATH="${MOUNT_BASE}/${DAOS_CONT}"

# Source location for models (HuggingFace Hub Cache)
# Primary source - ngetty's hub
MODEL_ROOT="${MODEL_ROOT:-/flare/ModCon/ngetty/huggingface/hub}"
# Secondary source - sandeep's hub (has walrus)
MODEL_ROOT_ALT="/flare/ModCon/sandeep/hub"

# PRISM model registry: models required for training
# Format: "model_id" (HuggingFace format, converted to models--org--name internally)
PRISM_MODELS=(
    # Backbone LLMs (1B and 7B variants)
    "allenai/OLMo-1B-0724-hf"
    "allenai/OLMo-7B-0724-hf"
    "allenai/Olmo-3-1025-7B"
    # Image encoder
    "google/siglip2-base-patch16-224"
    # Table encoder
    "google/tapas-base"
    # Time series encoder
    "Salesforce/moirai-2.0-R-small"
    # Geometry encoder (walrus) - in sandeep's hub
    "polymathic-ai/walrus"
    # Small LLM for auxiliary tasks
    "HuggingFaceTB/SmolLM2-360M-Instruct"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

check_daos_module() {
    if ! command -v daos &> /dev/null; then
        log_error "DAOS not loaded. Run: module use /soft/modulefiles && module load daos"
        exit 1
    fi
}

ensure_mounted() {
    if ! mount | grep -q "dfuse.*${DAOS_CONT}"; then
        log_info "Container not mounted, mounting first..."
        mount_container
    fi
}

# Helper to normalize model name to directory name
# allenai/OLMo-1B -> models--allenai--OLMo-1B
to_dir_name() {
    local name=$1
    if [[ "$name" == models--* ]]; then
        echo "$name"
    else
        echo "models--${name/\//--}"
    fi
}

# Helper to get friendly name from directory name
# models--allenai--OLMo-1B -> allenai/OLMo-1B
to_model_name() {
    local dir=$1
    local name=${dir#models--}
    echo "${name//--//}"
}

# Helper to find model source path (checks primary and alt sources)
# Returns the path if found, empty string if not
find_model_source() {
    local model_id=$1
    local dirname=$(to_dir_name "$model_id")
    
    if [ -d "$MODEL_ROOT/$dirname" ]; then
        echo "$MODEL_ROOT/$dirname"
    elif [ -d "$MODEL_ROOT_ALT/$dirname" ]; then
        echo "$MODEL_ROOT_ALT/$dirname"
    else
        echo ""
    fi
}

show_status() {
    check_daos_module
    
    log_section "DAOS Pool Status"
    daos pool query ${DAOS_POOL} 2>/dev/null || {
        log_error "Cannot query pool ${DAOS_POOL}. Check permissions."
        exit 1
    }
    
    log_section "Container List"
    daos cont list ${DAOS_POOL} 2>/dev/null || log_warn "No containers or cannot list"
    
    log_section "Mount Status"
    if mount | grep -q "dfuse.*${DAOS_CONT}"; then
        log_info "Container is mounted at: $(mount | grep dfuse | grep ${DAOS_CONT})"
    else
        log_warn "Container not currently mounted"
    fi

    if [ -d "${MOUNT_PATH}" ]; then
        log_section "Container Statistics"
        # Models are stored in hub/ subdirectory for HuggingFace compatibility
        local hub_path="${MOUNT_PATH}/hub"
        if [ -d "$hub_path" ]; then
            local count=$(ls -1d "${hub_path}"/models--* 2>/dev/null | wc -l)
            local size=$(du -sh "${hub_path}" 2>/dev/null | cut -f1)
            echo "Models stored: $count (in hub/)"
            echo "Total size: $size"
        else
            echo "Hub directory not yet created (run copy-all to populate)"
        fi
    fi
}

create_container() {
    check_daos_module
    
    log_info "Creating DAOS container: ${DAOS_POOL}/${DAOS_CONT}"
    
    # Check if container already exists
    if daos cont list ${DAOS_POOL} 2>/dev/null | grep -q "${DAOS_CONT}"; then
        log_warn "Container ${DAOS_CONT} already exists!"
        read -p "Do you want to continue with existing container? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return 0
    fi
    
    # Create with recommended settings for large file I/O (same as data container)
    # Models are often large binary files, so EC is appropriate
    daos container create --type=POSIX \
        --chunk-size=2097152 \
        --properties=rd_fac:3,ec_cell_sz:131072,cksum:crc32,srv_cksum:on \
        --file-oclass=EC_16P3GX \
        --dir-oclass=RP_4G1 \
        ${DAOS_POOL} ${DAOS_CONT}
    
    log_info "Container created successfully!"
}

mount_container() {
    check_daos_module
    
    # Check if already mounted
    if mount | grep -q "dfuse.*${DAOS_CONT}"; then
        log_warn "Container already mounted at $(mount | grep dfuse | grep ${DAOS_CONT} | awk '{print $3}')"
        return 0
    fi
    
    # Check for stale mount point
    if [ -d "${MOUNT_PATH}" ]; then
        if ! ls "${MOUNT_PATH}" &>/dev/null; then
            log_warn "Stale mount point detected, cleaning up..."
            fusermount3 -u "${MOUNT_PATH}" 2>/dev/null || true
            rmdir "${MOUNT_PATH}" 2>/dev/null || true
        fi
    fi
    
    log_info "Mounting container at: ${MOUNT_PATH}"
    
    mkdir -p "${MOUNT_PATH}"
    
    # Try dfuse mount
    if ! dfuse -m "${MOUNT_PATH}" --pool "${DAOS_POOL}" --cont "${DAOS_CONT}" 2>&1; then
        log_error "dfuse mount failed!"
        log_error "If running on UAN, you may need to run from a compute node:"
        log_error "  qsub -I -l select=1 -l walltime=1:00:00 -q debug -A ModCon -l filesystems=flare:home:daos_user_fs"
        return 1
    fi
    
    # Verify mount
    sleep 2
    if mount | grep -q "${MOUNT_PATH}" && ls "${MOUNT_PATH}" &>/dev/null; then
        log_info "Mount successful!"
    else
        log_error "Mount verification failed!"
        return 1
    fi
}

unmount_container() {
    if mount | grep -q "dfuse.*${MOUNT_PATH}"; then
        log_info "Unmounting: ${MOUNT_PATH}"
        fusermount3 -u "${MOUNT_PATH}"
        log_info "Unmounted successfully"
    else
        log_warn "Container not mounted at ${MOUNT_PATH}"
    fi
}

list_models() {
    log_section "PRISM Required Models"
    printf "%-50s %10s %15s %10s\n" "MODEL" "SIZE" "SOURCE" "ON_DAOS"
    echo "---------------------------------------------------------------------------------"
    
    local total_models=0
    local available=0
    local on_daos=0

    for model_id in "${PRISM_MODELS[@]}"; do
        local dirname=$(to_dir_name "$model_id")
        local size="N/A"
        local source_status="missing"
        local daos_status="no"
        
        # Check source (primary and alt)
        local src_path=$(find_model_source "$model_id")
        if [ -n "$src_path" ]; then
            size=$(du -sh "$src_path" 2>/dev/null | cut -f1)
            if [[ "$src_path" == "$MODEL_ROOT_ALT"* ]]; then
                source_status="alt"
            else
                source_status="yes"
            fi
            available=$((available + 1))
        fi
        
        # Check DAOS (in hub/ subdirectory for HF compatibility)
        if [ -d "${MOUNT_PATH}/hub/$dirname" ]; then
            daos_status="yes"
            on_daos=$((on_daos + 1))
        fi
        
        printf "%-50s %10s %15s %10s\n" "$model_id" "$size" "$source_status" "$daos_status"
        total_models=$((total_models + 1))
    done
    
    echo "---------------------------------------------------------------------------------"
    echo "Total: $total_models models, $available available in source, $on_daos on DAOS"
    echo ""
    echo "Source locations:"
    echo "  Primary: $MODEL_ROOT"
    echo "  Alt:     $MODEL_ROOT_ALT"
    echo "  DAOS:    $MOUNT_PATH/hub/"
    
    # Also show any extra models in source not in registry
    log_section "Other Models in Source (not in PRISM registry)"
    for dir in "$MODEL_ROOT"/models--*; do
        [ -d "$dir" ] || continue
        local dirname=$(basename "$dir")
        local name=$(to_model_name "$dirname")
        
        # Check if in registry
        local in_registry=0
        for model_id in "${PRISM_MODELS[@]}"; do
            if [ "$(to_dir_name "$model_id")" = "$dirname" ]; then
                in_registry=1
                break
            fi
        done
        
        if [ $in_registry -eq 0 ]; then
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            printf "  %-48s %10s\n" "$name" "$size"
        fi
    done
}

_copy_model_internal() {
    local name=$1
    local dirname=$(to_dir_name "$name")
    local src=$(find_model_source "$name")
    # Store in hub/ subdirectory for HuggingFace compatibility
    local dest="${MOUNT_PATH}/hub/${dirname}"

    if [ -z "$src" ] || [ ! -d "$src" ]; then
        log_error "Source model not found: $name"
        log_error "  Checked: $MODEL_ROOT/$dirname"
        log_error "  Checked: $MODEL_ROOT_ALT/$dirname"
        return 1
    fi

    log_info "Copying $name..."
    log_info "  Source: $src"
    log_info "  Dest:   $dest"

    # Ensure hub directory exists
    mkdir -p "${MOUNT_PATH}/hub"

    if [ -d "$dest" ]; then
        log_warn "  Destination already exists. Updating..."
    fi

    # Use rsync for robust copy of directory structure
    if ! command -v rsync &> /dev/null; then
        # Fallback to cp -r if rsync unavailable (unlikely on UAN)
        log_warn "rsync not found, using cp -r"
        cp -r "$src" "${MOUNT_PATH}/hub/"
    else
        rsync -aP --info=progress2 "$src/" "$dest/"
    fi
    
    log_info "  Copy complete."
}

copy_model() {
    local name=$1
    check_daos_module
    ensure_mounted
    
    _copy_model_internal "$name"
}

copy_all_models() {
    check_daos_module
    ensure_mounted
    
    log_section "Copy All PRISM Models to DAOS"
    echo "Source (primary): $MODEL_ROOT"
    echo "Source (alt):     $MODEL_ROOT_ALT"
    echo "Target:           $MOUNT_PATH/hub/"
    echo ""
    
    # Count available models
    local count=0
    local total_size=""
    echo "Models to copy:"
    for model_id in "${PRISM_MODELS[@]}"; do
        local src_path=$(find_model_source "$model_id")
        if [ -n "$src_path" ]; then
            local size=$(du -sh "$src_path" 2>/dev/null | cut -f1)
            local src_label="primary"
            [[ "$src_path" == "$MODEL_ROOT_ALT"* ]] && src_label="alt"
            printf "  %-50s %10s  (%s)\n" "$model_id" "$size" "$src_label"
            count=$((count + 1))
        else
            printf "  %-50s %10s\n" "$model_id" "(missing)"
        fi
    done
    
    echo ""
    echo "Found $count models available to copy."
    
    read -p "Proceed with copy? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Cancelled"
        return 0
    fi
    
    # Create hub directory
    mkdir -p "${MOUNT_PATH}/hub"
    
    local success=0
    local fail=0
    local skip=0
    
    for model_id in "${PRISM_MODELS[@]}"; do
        local src_path=$(find_model_source "$model_id")
        if [ -n "$src_path" ]; then
            if _copy_model_internal "$model_id"; then
                success=$((success + 1))
            else
                fail=$((fail + 1))
            fi
        else
            log_warn "Skipping $model_id (not found in any source)"
            skip=$((skip + 1))
        fi
        echo ""
    done
    
    log_section "Summary"
    echo "Copied: $success"
    echo "Skipped: $skip"
    echo "Failed: $fail"
}

show_help() {
    cat << EOF
DAOS Model Setup Script for PRISM

Usage: $0 <command> [args]

Commands:
  status              Show DAOS pool and container status
  create              Create the DAOS container (prism_models)
  mount               Mount container to $MOUNT_PATH
  unmount             Unmount container
  list                List PRISM required models and their status
  copy <model_id>     Copy specific model (e.g. allenai/OLMo-1B-0724-hf)
  copy-all            Copy all PRISM required models
  help                Show this help

PRISM Required Models:
EOF
    for model_id in "${PRISM_MODELS[@]}"; do
        echo "  - $model_id"
    done
    cat << EOF

Configuration:
  DAOS_POOL:      $DAOS_POOL
  DAOS_CONT:      $DAOS_CONT
  MODEL_ROOT:     $MODEL_ROOT
  MODEL_ROOT_ALT: $MODEL_ROOT_ALT
  MOUNT_PATH:     $MOUNT_PATH/hub/

Notes:
  - Models are stored in hub/ subdirectory for HuggingFace compatibility
  - Primary source is MODEL_ROOT, alt source (sandeep hub) used for walrus
  - The launcher (launch_aurora_daos.py) symlinks from DAOS instead of copying

Examples:
  # List all models and their status
  ./scripts/setup_daos_models.sh list

  # Copy all PRISM models to DAOS (run from compute node with DAOS access)
  qsub -I -l select=1 -l walltime=1:00:00 -q debug -A ModCon -l filesystems=flare:home:daos_user_fs
  ./scripts/setup_daos_models.sh copy-all

  # Copy a specific model
  ./scripts/setup_daos_models.sh copy allenai/OLMo-7B-0724-hf

EOF
}

# Main
case "${1:-help}" in
    status)
        show_status
        ;;
    create)
        create_container
        ;;
    mount)
        mount_container || exit 1
        ;;
    unmount)
        unmount_container
        ;;
    list)
        # Mount strictly optional for list, but needed to show status on DAOS
        # We try to mount if possible, but don't fail if we can't (e.g. no pool)
        if command -v daos &> /dev/null; then
             (mount | grep -q "dfuse.*${DAOS_CONT}") || mount_container || log_warn "Could not mount DAOS container, listing local only"
        fi
        list_models
        ;;
    copy)
        if [ -z "$2" ]; then
            log_error "Usage: $0 copy <model-name>"
            exit 1
        fi
        copy_model "$2"
        ;;
    copy-all)
        copy_all_models
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
