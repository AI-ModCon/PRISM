#!/bin/bash
# setup_daos_container.sh - Create and configure DAOS container for PRISM training data
#
# Usage:
#   ./scripts/setup_daos_container.sh [command] [args]
#
# Commands:
#   status              Show DAOS pool and container status
#   create              Create the DAOS container (one-time setup)
#   mount               Mount container to local path (UAN only)
#   unmount             Unmount container
#   list                List all available datasets with sizes
#   copy <name>         Copy a single dataset by name
#   copy-group <group>  Copy all datasets in a group
#   copy-all            Copy all registered datasets
#   migrate-pixmo       Move existing pixmo_cap to grouped structure
#   help                Show this help message
#
# Prerequisites:
#   - DAOS pool "AuroraGPT" must be allocated
#   - Run from login node (UAN) or compute node with daos_user_fs
#   - module load daos

set -e

# Configuration
DAOS_POOL="${DAOS_POOL:-AuroraGPT}"
DAOS_CONT="${DAOS_CONT:-prism_training_data}"
MOUNT_BASE="/tmp/${USER}/${DAOS_POOL}"
MOUNT_PATH="${MOUNT_BASE}/${DAOS_CONT}"

# Parallel copy settings
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"

# MPI parallel copy settings (for large datasets)
MPI_NODES="${MPI_NODES:-4}"
MPI_PPN="${MPI_PPN:-12}"  # ranks per node

# Dataset registry: "group:name:source_path"
# Groups: pixmo, s1mmalign, cosyn, nemotron
DATASET_REGISTRY=(
    # Pixmo datasets
    "pixmo:pixmo_cap:/flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset"
    "pixmo:pixmo_points:/flare/ModCon/ngetty/data/zone_a/pixmo_points_webdataset"
    "pixmo:pixmo_count:/flare/ModCon/ngetty/data/zone_a/pixmo_count_webdataset"
    
    # S1-MMAlign datasets (scientific papers)
    "s1mmalign:arxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/arxiv_webdataset"
    "s1mmalign:biorxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/biorxiv_webdataset"
    "s1mmalign:chemrxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/chemrxiv_webdataset"
    "s1mmalign:edrxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/edrxiv_webdataset"
    "s1mmalign:engrxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/engrxiv_webdataset"
    "s1mmalign:medrxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/medrxiv_webdataset"
    "s1mmalign:metarxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/metarxiv_webdataset"
    "s1mmalign:nature_comunication:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/nature_comunication_webdataset"
    "s1mmalign:psyarxiv:/flare/ModCon/ngetty/data/zone_a/s1mmalign_webdatasets/psyarxiv_webdataset"
    
    # CoSyn dataset
    "cosyn:cosyn_point:/flare/ModCon/ngetty/data/zone_a/cosyn_point_webdataset"
    
    # Nemotron datasets (13 complete - wiki + vision)
    "nemotron:wiki_de:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_de_webdataset"
    "nemotron:wiki_en:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_en_webdataset"
    "nemotron:wiki_es:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_es_webdataset"
    "nemotron:wiki_fr:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_fr_webdataset"
    "nemotron:wiki_it:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_it_webdataset"
    "nemotron:wiki_ja:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_ja_webdataset"
    "nemotron:wiki_ko:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_ko_webdataset"
    "nemotron:wiki_nl:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_nl_webdataset"
    "nemotron:wiki_pt:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_pt_webdataset"
    "nemotron:wiki_zh:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/wiki_zh_webdataset"
    "nemotron:sparsetables:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/sparsetables_webdataset"
    "nemotron:nights_cot:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/nights_cot_webdataset"
    "nemotron:plotqa_cot:/flare/ModCon/ngetty/data/zone_a/nemotron_webdatasets/plotqa_cot_webdataset"
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
        log_section "Container Contents"
        ls -la "${MOUNT_PATH}" 2>/dev/null || log_warn "Cannot list mount path"
        
        # Show grouped structure if it exists
        for group in pixmo s1mmalign cosyn nemotron; do
            if [ -d "${MOUNT_PATH}/${group}" ]; then
                echo ""
                echo "${group}/:"
                ls -1 "${MOUNT_PATH}/${group}" 2>/dev/null | head -10
                local count=$(ls -1 "${MOUNT_PATH}/${group}" 2>/dev/null | wc -l)
                [ $count -gt 10 ] && echo "  ... and $((count - 10)) more"
            fi
        done
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
    
    # Create with recommended settings for large file I/O
    # - chunk-size=2MB: Optimal for streaming reads
    # - rd_fac:3: 3-way redundancy for fault tolerance
    # - EC_16P3GX: Erasure coding across all servers (good for large shared files)
    # - RP_4G1: 4-way replication for directories (fast metadata)
    daos container create --type=POSIX \
        --chunk-size=2097152 \
        --properties=rd_fac:3,ec_cell_sz:131072,cksum:crc32,srv_cksum:on \
        --file-oclass=EC_16P3GX \
        --dir-oclass=RP_4G1 \
        ${DAOS_POOL} ${DAOS_CONT}
    
    log_info "Container created successfully!"
    
    log_section "Container Properties"
    daos container get-prop ${DAOS_POOL} ${DAOS_CONT}
}

mount_container() {
    check_daos_module
    
    # Check if already mounted
    if mount | grep -q "dfuse.*${DAOS_CONT}"; then
        log_warn "Container already mounted"
        mount | grep dfuse | grep ${DAOS_CONT}
        return 0
    fi
    
    log_info "Mounting container at: ${MOUNT_PATH}"
    
    mkdir -p "${MOUNT_PATH}"
    
    start-dfuse.sh -m "${MOUNT_PATH}" --pool ${DAOS_POOL} --cont ${DAOS_CONT}
    
    # Verify mount
    if mount | grep -q dfuse; then
        log_info "Mount successful!"
        echo "Mount path: ${MOUNT_PATH}"
        ls -la "${MOUNT_PATH}"
    else
        log_error "Mount failed!"
        exit 1
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

# List all available datasets with sizes
list_datasets() {
    log_section "Available Datasets"
    printf "%-12s %-22s %10s %10s %12s\n" "GROUP" "NAME" "SIZE" "SHARDS" "ON_DAOS"
    echo "------------------------------------------------------------------------"
    
    local total_size=0
    local total_shards=0
    local current_group=""
    
    for entry in "${DATASET_REGISTRY[@]}"; do
        IFS=':' read -r group name src <<< "$entry"
        
        # Add separator between groups
        if [ "$group" != "$current_group" ] && [ -n "$current_group" ]; then
            echo ""
        fi
        current_group="$group"
        
        if [ -d "$src" ]; then
            local size=$(du -sh "$src" 2>/dev/null | cut -f1)
            local shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
            
            # Check if on DAOS
            local dest="${MOUNT_PATH}/${group}/${name}_webdataset"
            local on_daos="no"
            if [ -d "$dest/shards" ]; then
                local daos_shards=$(ls "$dest/shards"/*.tar 2>/dev/null | wc -l)
                if [ "$daos_shards" -eq "$shards" ]; then
                    on_daos="yes"
                else
                    on_daos="partial"
                fi
            fi
            
            printf "%-12s %-22s %10s %10d %12s\n" "$group" "$name" "$size" "$shards" "$on_daos"
            total_shards=$((total_shards + shards))
        else
            printf "%-12s %-22s %10s %10s %12s\n" "$group" "$name" "N/A" "N/A" "missing"
        fi
    done
    
    echo "------------------------------------------------------------------------"
    echo "Total datasets: ${#DATASET_REGISTRY[@]}"
    echo "Total shards: $total_shards"
    
    # Show groups summary
    log_section "Groups Summary"
    for group in pixmo s1mmalign cosyn nemotron; do
        local count=0
        for entry in "${DATASET_REGISTRY[@]}"; do
            IFS=':' read -r g n s <<< "$entry"
            if [ "$g" = "$group" ] && [ -d "$s" ]; then
                count=$((count + 1))
            fi
        done
        echo "  $group: $count datasets"
    done
}

# Copy a single dataset (internal function, no confirmation)
_copy_dataset_internal() {
    local group=$1
    local name=$2
    local src=$3
    
    local dest="${MOUNT_PATH}/${group}/${name}_webdataset"
    
    # Check if already exists and complete
    if [ -d "${dest}/shards" ]; then
        local existing=$(ls "${dest}/shards"/*.tar 2>/dev/null | wc -l)
        local expected=$(ls "${src}/shards"/*.tar 2>/dev/null | wc -l)
        if [ "$existing" -eq "$expected" ]; then
            log_info "  $name: already complete ($existing shards), skipping"
            return 0
        else
            log_warn "  $name: partial copy ($existing/$expected shards), resuming"
        fi
    fi
    
    # Validate source
    if [ ! -d "$src" ]; then
        log_error "  $name: source not found: $src"
        return 1
    fi
    
    local src_shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
    local src_size=$(du -sh "$src" 2>/dev/null | cut -f1)
    
    log_info "  $name: copying $src_shards shards ($src_size)..."
    
    # Create destination
    mkdir -p "${dest}/shards" "${dest}/val_shards"
    
    # Copy manifest
    cp "$src/manifest.json" "$dest/" 2>/dev/null || true
    
    # Copy training shards in parallel
    ls "$src/shards"/*.tar 2>/dev/null | \
        xargs -P ${PARALLEL_JOBS} -I {} cp {} "${dest}/shards/"
    
    # Copy validation shards
    local val_count=$(ls "$src/val_shards"/*.tar 2>/dev/null | wc -l)
    if [ "$val_count" -gt 0 ]; then
        ls "$src/val_shards"/*.tar 2>/dev/null | \
            xargs -P ${PARALLEL_JOBS} -I {} cp {} "${dest}/val_shards/"
    fi
    
    # Verify
    local copied=$(ls "${dest}/shards"/*.tar 2>/dev/null | wc -l)
    if [ "$copied" -eq "$src_shards" ]; then
        log_info "  $name: complete ($copied shards)"
    else
        log_error "  $name: incomplete ($copied/$src_shards shards)"
        return 1
    fi
}

# Copy a single dataset by name (with confirmation)
copy_dataset() {
    local target_name=$1
    check_daos_module
    
    # Find dataset in registry
    local found=0
    local group="" name="" src=""
    for entry in "${DATASET_REGISTRY[@]}"; do
        IFS=':' read -r g n s <<< "$entry"
        if [ "$n" = "$target_name" ]; then
            group=$g; name=$n; src=$s; found=1; break
        fi
    done
    
    if [ $found -eq 0 ]; then
        log_error "Unknown dataset: $target_name"
        echo ""
        echo "Available datasets:"
        for entry in "${DATASET_REGISTRY[@]}"; do
            IFS=':' read -r g n s <<< "$entry"
            echo "  $n ($g)"
        done
        exit 1
    fi
    
    ensure_mounted
    
    # Show info and confirm
    local src_shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
    local src_size=$(du -sh "$src" 2>/dev/null | cut -f1)
    local dest="${MOUNT_PATH}/${group}/${name}_webdataset"
    
    log_section "Copy Dataset: $name"
    echo "Source: $src"
    echo "Destination: $dest"
    echo "Shards: $src_shards, Size: $src_size"
    echo ""
    
    read -p "Proceed with copy? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Cancelled"
        return 0
    fi
    
    _copy_dataset_internal "$group" "$name" "$src"
}

# Copy all datasets in a group
copy_group() {
    local target_group=$1
    check_daos_module
    
    # Validate group
    local valid_groups="pixmo s1mmalign cosyn nemotron"
    if ! echo "$valid_groups" | grep -qw "$target_group"; then
        log_error "Unknown group: $target_group"
        echo "Valid groups: $valid_groups"
        exit 1
    fi
    
    ensure_mounted
    
    # Collect datasets in group
    local datasets=()
    local total_shards=0
    
    log_section "Copy Group: $target_group"
    echo "Datasets to copy:"
    echo ""
    
    for entry in "${DATASET_REGISTRY[@]}"; do
        IFS=':' read -r group name src <<< "$entry"
        if [ "$group" = "$target_group" ] && [ -d "$src" ]; then
            local size=$(du -sh "$src" 2>/dev/null | cut -f1)
            local shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
            printf "  %-25s %10s %8d shards\n" "$name" "$size" "$shards"
            datasets+=("$entry")
            total_shards=$((total_shards + shards))
        fi
    done
    
    echo ""
    echo "Total: ${#datasets[@]} datasets, $total_shards shards"
    echo ""
    
    read -p "Proceed with copy? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Cancelled"
        return 0
    fi
    
    log_info "Starting copy with $PARALLEL_JOBS parallel jobs per dataset..."
    echo ""
    
    local copied=0
    local failed=0
    for entry in "${datasets[@]}"; do
        IFS=':' read -r group name src <<< "$entry"
        if _copy_dataset_internal "$group" "$name" "$src"; then
            copied=$((copied + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    echo ""
    log_section "Copy Complete"
    echo "Copied: $copied datasets"
    [ $failed -gt 0 ] && log_warn "Failed: $failed datasets"
}

# Copy all registered datasets
copy_all() {
    check_daos_module
    ensure_mounted
    
    # Collect all datasets
    local total_datasets=0
    local total_shards=0
    
    log_section "Copy All Datasets"
    echo ""
    
    for group in pixmo s1mmalign cosyn nemotron; do
        echo "$group:"
        for entry in "${DATASET_REGISTRY[@]}"; do
            IFS=':' read -r g name src <<< "$entry"
            if [ "$g" = "$group" ] && [ -d "$src" ]; then
                local size=$(du -sh "$src" 2>/dev/null | cut -f1)
                local shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
                printf "  %-25s %10s %8d shards\n" "$name" "$size" "$shards"
                total_datasets=$((total_datasets + 1))
                total_shards=$((total_shards + shards))
            fi
        done
        echo ""
    done
    
    echo "Total: $total_datasets datasets, $total_shards shards"
    echo ""
    
    read -p "Proceed with copying all datasets? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Cancelled"
        return 0
    fi
    
    log_info "Starting copy with $PARALLEL_JOBS parallel jobs per dataset..."
    echo ""
    
    local copied=0
    local failed=0
    local skipped=0
    
    for entry in "${DATASET_REGISTRY[@]}"; do
        IFS=':' read -r group name src <<< "$entry"
        if [ -d "$src" ]; then
            if _copy_dataset_internal "$group" "$name" "$src"; then
                copied=$((copied + 1))
            else
                failed=$((failed + 1))
            fi
        else
            skipped=$((skipped + 1))
        fi
    done
    
    echo ""
    log_section "Copy Complete"
    echo "Copied: $copied datasets"
    [ $skipped -gt 0 ] && log_warn "Skipped (source missing): $skipped datasets"
    [ $failed -gt 0 ] && log_error "Failed: $failed datasets"
}

# DSYNC-based copy (FASTEST for Lustre->DAOS)
# Uses DAOS-native dsync with MPI parallelization
copy_dsync() {
    local target_name=$1
    check_daos_module
    
    # Find dataset in registry
    local found=0
    local group="" name="" src=""
    for entry in "${DATASET_REGISTRY[@]}"; do
        IFS=':' read -r g n s <<< "$entry"
        if [ "$n" = "$target_name" ]; then
            group=$g; name=$n; src=$s; found=1; break
        fi
    done
    
    if [ $found -eq 0 ]; then
        log_error "Unknown dataset: $target_name"
        echo ""
        echo "Available datasets:"
        for entry in "${DATASET_REGISTRY[@]}"; do
            IFS=':' read -r g n s <<< "$entry"
            echo "  $n ($g)"
        done
        exit 1
    fi
    
    ensure_mounted
    
    local dest="${MOUNT_PATH}/${group}/${name}_webdataset"
    local src_shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
    local src_size=$(du -sh "$src" 2>/dev/null | cut -f1)
    
    log_section "DSYNC Parallel Copy: $name"
    echo "Source: $src"
    echo "Destination: $dest"
    echo "Shards: $src_shards, Size: $src_size"
    echo ""
    
    # Check if dsync is available
    if ! command -v dsync &> /dev/null; then
        log_error "dsync not found. Load daos module: module use /soft/modulefiles && module load daos"
        exit 1
    fi
    
    # Check if we're on a compute node with mpiexec
    if ! command -v mpiexec &> /dev/null; then
        log_error "mpiexec not found. Run this on a compute node allocation."
        log_info "Example: qsub -I -l select=${MPI_NODES} -l walltime=2:00:00 -q workq -A ModCon -l filesystems=flare:home:daos_user_fs"
        exit 1
    fi
    
    read -p "Proceed with dsync parallel copy? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Cancelled"
        return 0
    fi
    
    # Create destination directories
    mkdir -p "${dest}/shards" "${dest}/val_shards"
    
    # Copy manifest first
    cp "$src/manifest.json" "$dest/" 2>/dev/null || true
    
    log_info "Starting dsync parallel copy with $((MPI_NODES * MPI_PPN)) workers..."
    local start_time=$(date +%s)
    
    # dsync is DAOS-optimized and handles parallelization internally
    # Use mpiexec to run dsync across multiple nodes for maximum throughput
    mpiexec -n $((MPI_NODES * MPI_PPN)) -ppn ${MPI_PPN} \
        dsync --progress 10 \
              --bufsize 64MB \
              "$src/shards" "${dest}/shards"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Copy validation shards (usually small)
    local val_count=$(ls "$src/val_shards"/*.tar 2>/dev/null | wc -l)
    if [ "$val_count" -gt 0 ]; then
        log_info "Copying $val_count validation shards..."
        mpiexec -n $((MPI_NODES * MPI_PPN)) -ppn ${MPI_PPN} \
            dsync --progress 10 "$src/val_shards" "${dest}/val_shards"
    fi
    
    # Verify
    local copied=$(ls "${dest}/shards"/*.tar 2>/dev/null | wc -l)
    if [ "$copied" -eq "$src_shards" ]; then
        log_info "SUCCESS: Copied $copied shards in ${duration}s"
        # Calculate throughput
        local size_bytes=$(du -sb "$src" 2>/dev/null | cut -f1)
        local rate_gbps=$(echo "scale=2; $size_bytes / $duration / 1073741824" | bc 2>/dev/null || echo "N/A")
        log_info "Throughput: ~${rate_gbps} GB/s"
    else
        log_error "INCOMPLETE: $copied/$src_shards shards copied"
        return 1
    fi
}

# MPI-based parallel copy for large datasets (MUCH faster)
# Requires: running on compute node with mpiexec available
copy_mpi() {
    local target_name=$1
    check_daos_module
    
    # Find dataset in registry
    local found=0
    local group="" name="" src=""
    for entry in "${DATASET_REGISTRY[@]}"; do
        IFS=':' read -r g n s <<< "$entry"
        if [ "$n" = "$target_name" ]; then
            group=$g; name=$n; src=$s; found=1; break
        fi
    done
    
    if [ $found -eq 0 ]; then
        log_error "Unknown dataset: $target_name"
        echo ""
        echo "Available datasets:"
        for entry in "${DATASET_REGISTRY[@]}"; do
            IFS=':' read -r g n s <<< "$entry"
            echo "  $n ($g)"
        done
        exit 1
    fi
    
    ensure_mounted
    
    local dest="${MOUNT_PATH}/${group}/${name}_webdataset"
    local src_shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
    local src_size=$(du -sh "$src" 2>/dev/null | cut -f1)
    
    log_section "MPI Parallel Copy: $name"
    echo "Source: $src"
    echo "Destination: $dest"
    echo "Shards: $src_shards, Size: $src_size"
    echo "MPI Configuration: $MPI_NODES nodes x $MPI_PPN ranks = $((MPI_NODES * MPI_PPN)) total workers"
    echo ""
    
    # Check if we're on a compute node with mpiexec
    if ! command -v mpiexec &> /dev/null; then
        log_error "mpiexec not found. Run this on a compute node allocation."
        log_info "Example: qsub -I -l select=${MPI_NODES} -l walltime=2:00:00 -q workq -A ModCon -l filesystems=flare:home:daos_user_fs"
        exit 1
    fi
    
    read -p "Proceed with MPI parallel copy? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Cancelled"
        return 0
    fi
    
    # Create destination directories
    mkdir -p "${dest}/shards" "${dest}/val_shards"
    
    # Copy manifest first
    cp "$src/manifest.json" "$dest/" 2>/dev/null || true
    
    # Create file list for parallel copy
    local filelist=$(mktemp)
    ls "$src/shards"/*.tar > "$filelist"
    local total_files=$(wc -l < "$filelist")
    
    log_info "Starting MPI parallel copy of $total_files files..."
    local start_time=$(date +%s)
    
    # Use mpiexec to distribute copy across nodes
    # Each rank copies a subset of files based on its rank ID
    mpiexec -n $((MPI_NODES * MPI_PPN)) -ppn ${MPI_PPN} bash -c '
        FILELIST="'"$filelist"'"
        DEST="'"${dest}/shards"'"
        TOTAL=$(wc -l < "$FILELIST")
        RANK=${PALS_RANKID:-${PMI_RANK:-0}}
        SIZE=${PALS_SIZE:-${PMI_SIZE:-1}}
        
        # Calculate which files this rank should copy
        FILES_PER_RANK=$(( (TOTAL + SIZE - 1) / SIZE ))
        START=$(( RANK * FILES_PER_RANK + 1 ))
        END=$(( START + FILES_PER_RANK - 1 ))
        [ $END -gt $TOTAL ] && END=$TOTAL
        
        if [ $START -le $TOTAL ]; then
            sed -n "${START},${END}p" "$FILELIST" | while read f; do
                cp "$f" "$DEST/"
            done
        fi
    '
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    rm -f "$filelist"
    
    # Copy validation shards (usually small, single-node is fine)
    local val_count=$(ls "$src/val_shards"/*.tar 2>/dev/null | wc -l)
    if [ "$val_count" -gt 0 ]; then
        log_info "Copying $val_count validation shards..."
        ls "$src/val_shards"/*.tar 2>/dev/null | \
            xargs -P ${PARALLEL_JOBS} -I {} cp {} "${dest}/val_shards/"
    fi
    
    # Verify
    local copied=$(ls "${dest}/shards"/*.tar 2>/dev/null | wc -l)
    if [ "$copied" -eq "$src_shards" ]; then
        log_info "SUCCESS: Copied $copied shards in ${duration}s"
        local rate=$(echo "scale=2; $src_shards / $duration" | bc 2>/dev/null || echo "N/A")
        log_info "Rate: ~${rate} shards/sec"
    else
        log_error "INCOMPLETE: $copied/$src_shards shards copied"
        return 1
    fi
}

# Alternative: Use GNU parallel with SSH for multi-node (if MPI unavailable)
copy_parallel_ssh() {
    local target_name=$1
    local nodes_file="${2:-}"
    
    check_daos_module
    
    # Find dataset
    local found=0
    local group="" name="" src=""
    for entry in "${DATASET_REGISTRY[@]}"; do
        IFS=':' read -r g n s <<< "$entry"
        if [ "$n" = "$target_name" ]; then
            group=$g; name=$n; src=$s; found=1; break
        fi
    done
    
    if [ $found -eq 0 ]; then
        log_error "Unknown dataset: $target_name"
        exit 1
    fi
    
    ensure_mounted
    
    local dest="${MOUNT_PATH}/${group}/${name}_webdataset"
    local src_shards=$(ls "$src/shards"/*.tar 2>/dev/null | wc -l)
    local src_size=$(du -sh "$src" 2>/dev/null | cut -f1)
    
    log_section "GNU Parallel SSH Copy: $name"
    echo "Source: $src"  
    echo "Destination: $dest"
    echo "Shards: $src_shards, Size: $src_size"
    
    # Get node list from PBS or user-provided file
    if [ -n "$nodes_file" ] && [ -f "$nodes_file" ]; then
        local nodes=$(cat "$nodes_file" | tr '\n' ',' | sed 's/,$//')
    elif [ -n "$PBS_NODEFILE" ]; then
        local nodes=$(cat "$PBS_NODEFILE" | sort -u | tr '\n' ',' | sed 's/,$//')
        log_info "Using PBS nodes: $nodes"
    else
        log_error "No nodes available. Provide PBS_NODEFILE or a nodes file."
        log_info "Usage: $0 copy-parallel-ssh <dataset> [nodes_file]"
        exit 1
    fi
    
    if ! command -v parallel &> /dev/null; then
        log_error "GNU parallel not found. Install or use copy-mpi instead."
        exit 1
    fi
    
    read -p "Proceed? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 0
    fi
    
    mkdir -p "${dest}/shards" "${dest}/val_shards"
    cp "$src/manifest.json" "$dest/" 2>/dev/null || true
    
    log_info "Starting parallel copy across nodes: $nodes"
    local start_time=$(date +%s)
    
    # Use GNU parallel to distribute across nodes via SSH
    ls "$src/shards"/*.tar | \
        parallel --sshloginfile <(echo "$nodes" | tr ',' '\n') \
                 --jobs ${PARALLEL_JOBS} \
                 --bar \
                 "cp {} ${dest}/shards/"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Validation shards
    local val_count=$(ls "$src/val_shards"/*.tar 2>/dev/null | wc -l)
    if [ "$val_count" -gt 0 ]; then
        ls "$src/val_shards"/*.tar | xargs -P ${PARALLEL_JOBS} -I {} cp {} "${dest}/val_shards/"
    fi
    
    local copied=$(ls "${dest}/shards"/*.tar 2>/dev/null | wc -l)
    log_info "Copied $copied/$src_shards shards in ${duration}s"
}

# Migrate existing pixmo_cap from flat to grouped structure
migrate_pixmo() {
    check_daos_module
    ensure_mounted
    
    local old_path="${MOUNT_PATH}/pixmo_cap_webdataset"
    local new_path="${MOUNT_PATH}/pixmo/pixmo_cap_webdataset"
    
    log_section "Migrate pixmo_cap to Grouped Structure"
    echo "From: $old_path"
    echo "To:   $new_path"
    echo ""
    
    if [ ! -d "$old_path" ]; then
        log_warn "Old path does not exist: $old_path"
        
        # Check if already migrated
        if [ -d "$new_path" ]; then
            log_info "Already at new location: $new_path"
            ls -la "$new_path"
        fi
        return 0
    fi
    
    if [ -d "$new_path" ]; then
        log_error "New path already exists: $new_path"
        log_error "Please remove one of them manually"
        exit 1
    fi
    
    local shards=$(ls "$old_path/shards"/*.tar 2>/dev/null | wc -l)
    local size=$(du -sh "$old_path" 2>/dev/null | cut -f1)
    echo "Shards: $shards, Size: $size"
    echo ""
    
    read -p "Proceed with migration? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Cancelled"
        return 0
    fi
    
    log_info "Creating group directory..."
    mkdir -p "${MOUNT_PATH}/pixmo"
    
    log_info "Moving pixmo_cap_webdataset..."
    mv "$old_path" "$new_path"
    
    log_info "Migration complete!"
    ls -la "$new_path"
}

show_help() {
    cat << EOF
DAOS Container Setup for PRISM Training

Usage: $0 <command> [args]

Commands:
  status              Show DAOS pool and container status
  create              Create the DAOS container (one-time setup)
  mount               Mount container to local path (UAN only)
  unmount             Unmount container
  
  list                List all available datasets with sizes and DAOS status
  copy <name>         Copy a single dataset by name (single-node, slow)
  copy-group <group>  Copy all datasets in a group
  copy-all            Copy all registered datasets
  
  copy-mpi <name>     FAST: MPI parallel copy across multiple nodes
  copy-dsync <name>   FASTEST: DAOS dsync with MPI (Lustre->DAOS optimized)
  copy-parallel-ssh   Alternative: GNU parallel with SSH
  
  migrate-pixmo       Move existing pixmo_cap to grouped structure
  
  help                Show this help message

Dataset Groups:
  pixmo       Pixmo datasets (1 dataset)
  s1mmalign   Scientific paper datasets (8 datasets, ~262 GB)
  cosyn       CoSyn point dataset (1 dataset, ~12 GB)
  nemotron    Nemotron VLM datasets (13 datasets, ~353 GB)

Environment Variables:
  DAOS_POOL       Pool name (default: AuroraGPT)
  DAOS_CONT       Container name (default: prism_training_data)
  PARALLEL_JOBS   Parallel copy jobs per node (default: 8)
  MPI_NODES       Number of nodes for MPI copy (default: 4)
  MPI_PPN         Ranks per node for MPI copy (default: 12)

Examples:
  # List all available datasets
  $0 list

  # Copy a single dataset
  $0 copy pixmo_cap
  $0 copy biorxiv

  # Copy an entire group
  $0 copy-group s1mmalign
  $0 copy-group nemotron

  # Copy everything (run in screen session)
  screen -S daos_copy
  $0 copy-all

  # FAST: MPI parallel copy for large datasets
  # First, get a compute node allocation:
  qsub -I -l select=4 -l walltime=2:00:00 -q workq -A ModCon -l filesystems=flare:home:daos_user_fs
  
  # Then run MPI copy:
  MPI_NODES=4 MPI_PPN=12 $0 copy-mpi arxiv

  # FASTEST: dsync for Lustre->DAOS (RECOMMENDED for large datasets like arxiv)
  MPI_NODES=4 MPI_PPN=12 $0 copy-dsync arxiv

  # Migrate existing pixmo_cap to grouped structure
  $0 migrate-pixmo

DAOS Structure (after copy):
  /prism_training_data/
    ├── pixmo/
    │   └── pixmo_cap_webdataset/
    ├── s1mmalign/
    │   ├── biorxiv_webdataset/
    │   ├── chemrxiv_webdataset/
    │   └── ...
    ├── cosyn/
    │   └── cosyn_point_webdataset/
    └── nemotron/
        ├── wiki_en_webdataset/
        ├── sparsetables_webdataset/
        └── ...

Notes:
  - Container creation only needs to be done once
  - On compute nodes, use launch-dfuse.sh (handled by job script)
  - Interception library (libpil4dfs.so) improves I/O performance
  - Parallel jobs can be adjusted via PARALLEL_JOBS environment variable
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
        mount_container
        ;;
    unmount)
        unmount_container
        ;;
    list)
        list_datasets
        ;;
    copy)
        if [ -z "$2" ]; then
            log_error "Usage: $0 copy <dataset-name>"
            echo ""
            echo "Available datasets:"
            for entry in "${DATASET_REGISTRY[@]}"; do
                IFS=':' read -r g n s <<< "$entry"
                echo "  $n ($g)"
            done
            exit 1
        fi
        copy_dataset "$2"
        ;;
    copy-group)
        if [ -z "$2" ]; then
            log_error "Usage: $0 copy-group <group-name>"
            echo "Groups: pixmo, s1mmalign, cosyn, nemotron"
            exit 1
        fi
        copy_group "$2"
        ;;
    copy-all)
        copy_all
        ;;
    copy-mpi)
        if [ -z "$2" ]; then
            log_error "Usage: $0 copy-mpi <dataset-name>"
            echo ""
            echo "MPI parallel copy for large datasets (run on compute node)"
            echo "Environment variables:"
            echo "  MPI_NODES=$MPI_NODES (number of nodes)"
            echo "  MPI_PPN=$MPI_PPN (ranks per node)"
            exit 1
        fi
        copy_mpi "$2"
        ;;
    copy-dsync)
        if [ -z "$2" ]; then
            log_error "Usage: $0 copy-dsync <dataset-name>"
            echo ""
            echo "DSYNC parallel copy - FASTEST for Lustre->DAOS transfers"
            echo "Uses DAOS-native dsync with MPI parallelization"
            echo ""
            echo "Environment variables:"
            echo "  MPI_NODES=$MPI_NODES (number of nodes)"
            echo "  MPI_PPN=$MPI_PPN (ranks per node)"
            exit 1
        fi
        copy_dsync "$2"
        ;;
    copy-parallel-ssh)
        if [ -z "$2" ]; then
            log_error "Usage: $0 copy-parallel-ssh <dataset-name> [nodes_file]"
            exit 1
        fi
        copy_parallel_ssh "$2" "${3:-}"
        ;;
    migrate-pixmo)
        migrate_pixmo
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
