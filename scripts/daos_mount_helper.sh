#!/bin/bash
# daos_mount_helper.sh - Unified DAOS mounting for PRISM training
#
# This script provides consistent DAOS mounting for both interactive and batch jobs.
# It should be sourced by other scripts or called directly.
#
# Usage:
#   source scripts/daos_mount_helper.sh
#   mount_daos_containers  # Mount both data and models containers
#
# Or directly:
#   ./scripts/daos_mount_helper.sh mount       # Mount on current node
#   ./scripts/daos_mount_helper.sh mount-all   # Mount on all nodes (multi-node jobs)
#   ./scripts/daos_mount_helper.sh status      # Check mount status
#   ./scripts/daos_mount_helper.sh unmount     # Unmount on current node
#
# Environment Variables (set these before sourcing/calling):
#   DAOS_POOL       - Pool name (default: AuroraGPT)
#   DAOS_DATA_CONT  - Data container (default: prism_training_data)
#   DAOS_MODELS_CONT - Models container (default: prism_models)
#
# Output Variables (set after successful mount):
#   DAOS_MOUNT      - Path to data container mount
#   DAOS_MODELS_MOUNT - Path to models container mount
#   DAOS_MODELS_HUB - Path to HuggingFace hub within models container

set -e

# ============================================================================
# Configuration
# ============================================================================

export DAOS_POOL="${DAOS_POOL:-AuroraGPT}"
export DAOS_DATA_CONT="${DAOS_DATA_CONT:-prism_training_data}"
export DAOS_MODELS_CONT="${DAOS_MODELS_CONT:-prism_models}"

# Mount paths - consistent for all users and all job types
# Using /tmp/${USER}/${DAOS_POOL}/ structure for clarity
export DAOS_MOUNT_BASE="/tmp/${USER}/${DAOS_POOL}"
export DAOS_MOUNT="${DAOS_MOUNT_BASE}/${DAOS_DATA_CONT}"
export DAOS_MODELS_MOUNT="${DAOS_MOUNT_BASE}/${DAOS_MODELS_CONT}"
export DAOS_MODELS_HUB="${DAOS_MODELS_MOUNT}/hub"

# Fallback paths for models (Lustre)
export FALLBACK_HF_HOME="/flare/ModCon/ngetty/huggingface/hub"
export FALLBACK_HF_HOME_ALT="/flare/ModCon/sandeep/hub"

# Colors
_RED='\033[0;31m'
_GREEN='\033[0;32m'
_YELLOW='\033[1;33m'
_NC='\033[0m'

_log_info() { echo -e "${_GREEN}[DAOS]${_NC} $1"; }
_log_warn() { echo -e "${_YELLOW}[DAOS]${_NC} $1"; }
_log_error() { echo -e "${_RED}[DAOS]${_NC} $1"; }

# ============================================================================
# Core Functions
# ============================================================================

# Check if DAOS module is loaded
check_daos_module() {
    if ! command -v daos &> /dev/null; then
        _log_error "DAOS not loaded. Run: module use /soft/modulefiles && module load daos"
        return 1
    fi
    return 0
}

# Check if a container is already mounted
is_mounted() {
    local mount_path=$1
    if mount | grep -q "dfuse.*${mount_path}"; then
        return 0
    fi
    # Also check if directory exists and is accessible
    if [ -d "$mount_path" ] && ls "$mount_path" &>/dev/null 2>&1; then
        # Check if it has expected content
        if [ -n "$(ls -A "$mount_path" 2>/dev/null)" ]; then
            return 0
        fi
    fi
    return 1
}

# Mount the data container
mount_data_container() {
    if is_mounted "$DAOS_MOUNT"; then
        _log_info "Data container already mounted at $DAOS_MOUNT"
        return 0
    fi
    
    _log_info "Mounting data container at $DAOS_MOUNT..."
    mkdir -p "$DAOS_MOUNT"
    
    # Use start-dfuse.sh if available (Aurora standard), otherwise dfuse directly
    if command -v start-dfuse.sh &> /dev/null; then
        start-dfuse.sh -m "$DAOS_MOUNT" --pool "$DAOS_POOL" --cont "$DAOS_DATA_CONT"
    else
        dfuse -m "$DAOS_MOUNT" --pool "$DAOS_POOL" --cont "$DAOS_DATA_CONT" --disable-wb-cache &
        sleep 3
    fi
    
    # Verify
    if is_mounted "$DAOS_MOUNT"; then
        _log_info "Data container mounted successfully"
        return 0
    else
        _log_error "Failed to mount data container"
        return 1
    fi
}

# Mount the models container
mount_models_container() {
    if is_mounted "$DAOS_MODELS_MOUNT"; then
        _log_info "Models container already mounted at $DAOS_MODELS_MOUNT"
        return 0
    fi
    
    _log_info "Mounting models container at $DAOS_MODELS_MOUNT..."
    mkdir -p "$DAOS_MODELS_MOUNT"
    
    if command -v start-dfuse.sh &> /dev/null; then
        start-dfuse.sh -m "$DAOS_MODELS_MOUNT" --pool "$DAOS_POOL" --cont "$DAOS_MODELS_CONT"
    else
        dfuse -m "$DAOS_MODELS_MOUNT" --pool "$DAOS_POOL" --cont "$DAOS_MODELS_CONT" --disable-wb-cache &
        sleep 3
    fi
    
    # Verify
    if is_mounted "$DAOS_MODELS_MOUNT"; then
        _log_info "Models container mounted successfully"
        return 0
    else
        _log_warn "Models container not available - will fall back to Lustre"
        return 1
    fi
}

# Mount both containers
mount_daos_containers() {
    check_daos_module || return 1
    
    local data_ok=0
    local models_ok=0
    
    mount_data_container && data_ok=1
    mount_models_container && models_ok=1
    
    # Data container is required, models is optional
    if [ $data_ok -eq 0 ]; then
        _log_error "Data container mount failed - cannot proceed"
        return 1
    fi
    
    # Export status for job scripts
    export DAOS_DATA_AVAILABLE=1
    if [ $models_ok -eq 1 ]; then
        export DAOS_MODELS_AVAILABLE=1
    else
        export DAOS_MODELS_AVAILABLE=0
    fi
    
    # Show status
    _log_info "DAOS mounts ready:"
    echo "  Data:   $DAOS_MOUNT"
    echo "  Models: $DAOS_MODELS_MOUNT (available: $DAOS_MODELS_AVAILABLE)"
    echo "  Hub:    $DAOS_MODELS_HUB"
    
    return 0
}

# Mount on all nodes in a multi-node job
mount_all_nodes() {
    check_daos_module || return 1
    
    local hosts=""
    
    # Get hosts from various sources
    if [ -n "$EXPLICIT_HOSTS" ]; then
        hosts="$EXPLICIT_HOSTS"
    elif [ -n "$PBS_NODEFILE" ] && [ -f "$PBS_NODEFILE" ]; then
        hosts=$(cat "$PBS_NODEFILE" | sort -u | tr '\n' ',' | sed 's/,$//')
    else
        _log_warn "No multi-node context, mounting locally only"
        mount_daos_containers
        return $?
    fi
    
    _log_info "Mounting DAOS on all nodes: $hosts"
    
    # Script to run on each node
    local mount_script=$(cat << 'SCRIPT'
module use /soft/modulefiles 2>/dev/null
module load daos 2>/dev/null

DAOS_POOL="${DAOS_POOL:-AuroraGPT}"
DAOS_MOUNT_BASE="/tmp/${USER}/${DAOS_POOL}"
DAOS_DATA_MOUNT="${DAOS_MOUNT_BASE}/prism_training_data"
DAOS_MODELS_MOUNT="${DAOS_MOUNT_BASE}/prism_models"

# Mount data container if not already mounted
if ! mount | grep -q "dfuse.*${DAOS_DATA_MOUNT}"; then
    mkdir -p "$DAOS_DATA_MOUNT"
    dfuse -m "$DAOS_DATA_MOUNT" --pool "$DAOS_POOL" --cont prism_training_data --disable-wb-cache &
fi

# Mount models container if not already mounted
if ! mount | grep -q "dfuse.*${DAOS_MODELS_MOUNT}"; then
    mkdir -p "$DAOS_MODELS_MOUNT"
    dfuse -m "$DAOS_MODELS_MOUNT" --pool "$DAOS_POOL" --cont prism_models --disable-wb-cache &
fi

sleep 3

# Verify
if [ -d "$DAOS_DATA_MOUNT" ] && ls "$DAOS_DATA_MOUNT" &>/dev/null; then
    echo "Data mount OK on $(hostname)"
else
    echo "Data mount FAILED on $(hostname)" >&2
fi
SCRIPT
)
    
    # Run on each host
    for host in $(echo "$hosts" | tr ',' ' '); do
        _log_info "Mounting on $host..."
        ssh "$host" "bash -c '$mount_script'" 2>&1 | sed "s/^/  [$host] /"
    done
    
    # Wait for mounts to stabilize
    sleep 2
    
    # Verify on current node
    mount_daos_containers
}

# Unmount containers on current node
unmount_daos_containers() {
    _log_info "Unmounting DAOS containers..."
    
    if mount | grep -q "dfuse.*${DAOS_MOUNT}"; then
        fusermount3 -u "$DAOS_MOUNT" 2>/dev/null || true
        _log_info "Unmounted data container"
    fi
    
    if mount | grep -q "dfuse.*${DAOS_MODELS_MOUNT}"; then
        fusermount3 -u "$DAOS_MODELS_MOUNT" 2>/dev/null || true
        _log_info "Unmounted models container"
    fi
}

# Show status
show_status() {
    echo "DAOS Mount Status:"
    echo "  Pool:           $DAOS_POOL"
    echo "  Data Container: $DAOS_DATA_CONT"
    echo "  Data Mount:     $DAOS_MOUNT"
    
    if is_mounted "$DAOS_MOUNT"; then
        echo "    Status:       MOUNTED"
        echo "    Contents:     $(ls "$DAOS_MOUNT" 2>/dev/null | head -5 | tr '\n' ' ')"
    else
        echo "    Status:       NOT MOUNTED"
    fi
    
    echo ""
    echo "  Models Container: $DAOS_MODELS_CONT"
    echo "  Models Mount:     $DAOS_MODELS_MOUNT"
    
    if is_mounted "$DAOS_MODELS_MOUNT"; then
        echo "    Status:       MOUNTED"
        if [ -d "$DAOS_MODELS_HUB" ]; then
            echo "    Hub Models:   $(ls "$DAOS_MODELS_HUB" 2>/dev/null | wc -l)"
        fi
    else
        echo "    Status:       NOT MOUNTED"
    fi
}

# Stage models from DAOS to local /tmp (for HuggingFace compatibility)
# This creates symlinks from /tmp/huggingface/hub to DAOS models
stage_models_from_daos() {
    local local_hf_home="${1:-/tmp/huggingface/hub}"
    
    mkdir -p "$local_hf_home"
    
    # List of models to stage
    local models=(
        "models--allenai--OLMo-1B-0724-hf"
        "models--allenai--OLMo-7B-0724-hf"
        "models--google--siglip2-base-patch16-224"
        "models--google--tapas-base"
        "models--Salesforce--moirai-2.0-R-small"
        "models--polymathic-ai--walrus"
        "models--HuggingFaceTB--SmolLM2-360M-Instruct"
    )
    
    for model in "${models[@]}"; do
        if [ -d "$DAOS_MODELS_HUB/$model" ] && [ ! -e "$local_hf_home/$model" ]; then
            ln -sf "$DAOS_MODELS_HUB/$model" "$local_hf_home/$model"
            _log_info "Linked $model from DAOS"
        elif [ -d "$FALLBACK_HF_HOME/$model" ] && [ ! -e "$local_hf_home/$model" ]; then
            _log_warn "$model not on DAOS, copying from Lustre..."
            cp -r "$FALLBACK_HF_HOME/$model" "$local_hf_home/"
        elif [ -d "$FALLBACK_HF_HOME_ALT/$model" ] && [ ! -e "$local_hf_home/$model" ]; then
            _log_warn "$model not on DAOS, copying from Lustre (alt)..."
            cp -r "$FALLBACK_HF_HOME_ALT/$model" "$local_hf_home/"
        fi
    done
}

# ============================================================================
# Main (when called directly)
# ============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-status}" in
        mount)
            mount_daos_containers
            ;;
        mount-all)
            mount_all_nodes
            ;;
        unmount)
            unmount_daos_containers
            ;;
        status)
            show_status
            ;;
        stage-models)
            stage_models_from_daos "${2:-/tmp/huggingface/hub}"
            ;;
        help|--help|-h)
            cat << EOF
DAOS Mount Helper for PRISM Training

Usage: $0 <command>

Commands:
  mount         Mount DAOS containers on current node
  mount-all     Mount on all nodes (multi-node jobs)
  unmount       Unmount containers on current node
  status        Show mount status
  stage-models  Create symlinks from /tmp to DAOS models

Environment Variables:
  DAOS_POOL        Pool name (default: AuroraGPT)
  DAOS_DATA_CONT   Data container (default: prism_training_data)
  DAOS_MODELS_CONT Models container (default: prism_models)
  EXPLICIT_HOSTS   Comma-separated host list for mount-all

Output Paths (after mount):
  DAOS_MOUNT         $DAOS_MOUNT
  DAOS_MODELS_MOUNT  $DAOS_MODELS_MOUNT
  DAOS_MODELS_HUB    $DAOS_MODELS_HUB

Example:
  # Interactive job setup
  qsub -I -l select=2 -l walltime=1:00:00 -q debug -A ModCon \
       -l filesystems=flare:home:daos_user_fs
  
  module use /soft/modulefiles && module load daos
  ./scripts/daos_mount_helper.sh mount
  
  # Multi-node job
  EXPLICIT_HOSTS=node1,node2 ./scripts/daos_mount_helper.sh mount-all
EOF
            ;;
        *)
            echo "Unknown command: $1"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
fi
