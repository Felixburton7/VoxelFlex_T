#!/bin/bash

# ==========================================================
# Create VoxelFlex Project Context File Script
# ==========================================================
# This script analyzes the current VoxelFlex project directory
# and generates a comprehensive context file describing its
# goals, data, workflow (Metadata-Only Preprocessing plan),
# scale, compute environment, structure, and source code content.
# It includes snippets (head -n 10) for potentially large input
# files like domain lists.
#
# Run this script from the root directory of the VoxelFlex project.

OUTPUT_FILE="voxelflex_context.txt"

# --- Basic Checks ---
if [ ! -d "src/voxelflex" ]; then
  echo "ERROR: 'src/voxelflex' directory not found."
  echo "Please run this script from the root directory of the VoxelFlex project."
  exit 1
fi
if [ ! -f "pyproject.toml" ]; then
  echo "ERROR: 'pyproject.toml' not found."
  echo "Please run this script from the root directory of the VoxelFlex project."
  exit 1
fi

echo "Generating project context file: $OUTPUT_FILE ..."

# Clear existing file
rm -f "$OUTPUT_FILE"

# --- Header ---
echo "==========================================================" >> "$OUTPUT_FILE"
echo "        VoxelFlex (Temperature-Aware) Project Context     " >> "$OUTPUT_FILE"
echo "     (Workflow: Metadata Preprocessing / On-Demand HDF5)    " >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# --- Project Goal & Workflow Overview ---
echo "Project Goal & Workflow Overview:" >> "$OUTPUT_FILE"
echo "---------------------------------" >> "$OUTPUT_FILE"
# *** FIX: Use quoted EOF to prevent shell interpretation ***
cat << 'EOF' >> "$OUTPUT_FILE"
The primary goal of the 'voxelflex' package is to predict per-residue protein flexibility (Root Mean Square Fluctuation, RMSF) using 3D structural data (voxels) while incorporating temperature awareness. A single trained model should be capable of predicting RMSF for a given protein structure at a specific input temperature, intended as a component for a larger multi-modal prediction framework (OmniFlex).

This version implements the **'Metadata Only Preprocessing'** workflow to address previous issues with I/O bottlenecks, excessive intermediate file creation, filesystem instability, and memory constraints during preprocessing attempts.

**Core Workflow:**

1.  **`preprocess` Command (Metadata Only):**
    *   **Input:** Raw HDF5 voxel file (for keys/structure), aggregated multi-temperature RMSF CSV, domain split files (`.txt`).
    *   **Process:** Loads RMSF data, HDF5 keys, creates mappings, loads domain splits, filters them, generates a **master list** of all valid samples across splits `(hdf5_domain_id, resid_str, resid_int, raw_temp, target_rmsf, split)`. This involves checking HDF5 residue keys against the RMSF lookup but **does not load or process voxel data**. Calculates temperature scaler based on *training* samples and saves to `temp_scaling_params.json` in the run's output directory.
    *   **Output:** Saves the master sample list to a single file (`master_samples.parquet` or `.csv`) in `processed_dir`. Saves `failed_preprocess_domains.txt` in the run output directory. Minimal resource usage.

2.  **`train` Command (On-Demand HDF5 Loading):**
    *   **Input:** Config file, `master_samples.parquet` (or `.csv`), `temp_scaling_params.json`, raw HDF5 voxel file.
    *   **Process:** Checks for master samples file (runs `preprocess` if missing/forced). Instantiates `VoxelDataset`. `DataLoader` with `num_workers > 0` coordinates workers. **Workers** execute `VoxelDataset.__getitem__` -> read required HDF5 voxel data **on-demand**, process (cast/transpose), scale temp, return tensors. Trains model.
    *   **Output:** Model checkpoints (`.pt`), history (`.json`), logs.

3.  **`predict` Command:** Loads model/scaler. Loads raw voxel data **on-demand** from HDF5, processes, runs inference. Saves predictions CSV.

4.  **`evaluate` Command:** Merges predictions CSV and ground truth RMSF CSV. Calculates metrics. If permutation importance enabled, loads raw voxel data **on-demand** from HDF5. Saves metrics JSON.

5.  **`visualize` Command:** Loads predictions CSV, ground truth RMSF CSV, optional history. Generates plots. Saves plots and optional data.

This workflow minimizes preprocessing load/storage, shifting HDF5 interaction to parallel DataLoader workers during training.
EOF
echo "" >> "$OUTPUT_FILE"


# --- Dataset Scale ---
echo "Dataset Scale (Approximate):" >> "$OUTPUT_FILE"
echo "----------------------------" >> "$OUTPUT_FILE"
# *** FIX: Use quoted EOF ***
cat << 'EOF' >> "$OUTPUT_FILE"
*   **Domains:** ~5,400 unique domain IDs in the source HDF5 file. (~5,378 found relevant across provided splits).
*   **RMSF Data:** > 3.3 million rows in the aggregated CSV file (representing multiple temperatures per residue).
*   **Samples Generated:** The preprocessing step generates > 3.3 million individual samples (residue@temperature points) stored in the `master_samples.parquet` file.
*   **Voxel Data Size:** Individual processed voxel arrays (float32, 5x21x21x21) are ~180KB each. The raw HDF5 file is likely hundreds of GBs.
*   **Intermediate Data:** The `master_samples.parquet` file (metadata only) is relatively small (MBs to low GBs). **This workflow avoids large intermediate voxel storage.**
EOF
echo "" >> "$OUTPUT_FILE"


# --- Compute Environment Specifications ---
echo "Compute Environment Specifications:" >> "$OUTPUT_FILE"
echo "---------------------------------" >> "$OUTPUT_FILE"
# *** FIX: Use quoted EOF ***
cat << 'EOF' >> "$OUTPUT_FILE"
*   **CPU:** 36 Cores (Based on user info and `htop` output)
*   **RAM:** ~62.6 GB Total System Memory
*   **GPU:** 1x NVIDIA Quadro RTX 8000 (49152 MiB / ~47.5 GB VRAM)
*   **CUDA Version:** 12.2 (from `nvidia-smi`)
*   **NVIDIA Driver Version:** 535.183.01 (from `nvidia-smi`)
*   **Operating System:** Linux (Implied)
*   **Filesystem Mount:** `/home/s_felix` located on `/dev/sdc2` (Previously experienced read-only remount issues under heavy I/O load).
EOF
echo "" >> "$OUTPUT_FILE"


# --- Input Data Formats (Expected) ---
echo "Input Data Formats (Expected):" >> "$OUTPUT_FILE"
echo "------------------------------" >> "$OUTPUT_FILE"
# *** FIX: Use quoted EOF ***
cat << 'EOF' >> "$OUTPUT_FILE"
1.  **Voxel Data (HDF5):**
    *   Path: `input.voxel_file` in config.
    *   Format: HDF5 (`.hdf5`).
    *   Structure: `DomainID` -> `ChainID` -> `ResidueID` (string digit) -> HDF5 Dataset.
    *   Voxel Dataset: Expected `bool`, shape `(21, 21, 21, 5)`. Processed on-demand to `float32`, shape `(5, 21, 21, 21)`.

2.  **Aggregated RMSF Data (CSV):**
    *   Path: `input.aggregated_rmsf_file` in config.
    *   Format: CSV (`.csv`).
    *   Required Columns: `domain_id`, `resid`, `resname`, `temperature_feature`, `target_rmsf`.
    *   Optional Columns: `relative_accessibility`, `dssp`, `secondary_structure_encoded`.

3.  **Domain Split Files (.txt):**
    *   Paths: `input.train_split_file`, etc. in config.
    *   Format: Plain text (`.txt`), one HDF5 `DomainID` per line.
EOF
echo "" >> "$OUTPUT_FILE"


# --- Output Data Formats (Expected) ---
echo "Output Data Formats (Expected):" >> "$OUTPUT_FILE"
echo "-------------------------------" >> "$OUTPUT_FILE"
# *** FIX: Use quoted EOF ***
cat << 'EOF' >> "$OUTPUT_FILE"
Outputs saved within `outputs/<run_name>/`.

1.  **From `preprocess`:**
    *   `input_data/processed/master_samples.parquet` (or `.csv`): Single file with sample metadata. **NO VOXELS.**
    *   `outputs/<run_name>/models/temp_scaling_params.json`: Scaler min/max.
    *   `outputs/<run_name>/failed_preprocess_domains.txt`: Domains failing initial checks.

2.  **From `train`:**
    *   `outputs/<run_name>/models/*.pt`: Model checkpoints.
    *   `outputs/<run_name>/training_history.json`: Epoch metrics.
    *   `outputs/<run_name>/logs/voxelflex.log`: Detailed log.

3.  **From `predict`:**
    *   `outputs/<run_name>/metrics/predictions_*.csv`: Predictions CSV.

4.  **From `evaluate`:**
    *   `outputs/<run_name>/metrics/evaluation_metrics_*.json`: Metrics results JSON.

5.  **From `visualize`:**
    *   `outputs/<run_name>/visualizations/*.png`: Plot images.
    *   `outputs/<run_name>/visualizations/*_data.csv` (Optional): Plot data CSVs.
EOF
echo "" >> "$OUTPUT_FILE"


# --- Folder Structure ---
echo "Project Folder Structure:" >> "$OUTPUT_FILE"
echo "-------------------------" >> "$OUTPUT_FILE"
echo "(Showing relative paths from project root)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
if command -v tree &> /dev/null; then
  tree -L 3 -I '.git*|__pycache__|*.egg-info|.venv|env|venv|outputs|snap' >> "$OUTPUT_FILE"
else
  echo "(tree command not found, using find...)" >> "$OUTPUT_FILE"
  find . -not \( \
      -path './.git*' -o \
      -path './__pycache__*' -o \
      -path './*.egg-info*' -o \
      -path './.venv*' -o \
      -path './venv*' -o \
      -path './env*' -o \
      -path './outputs*' -o \
      -path './snap' -o \
      -name "$OUTPUT_FILE" \
      \) -print | sed -e 's;[^/]*/;|____;g;s;____|; |;s;[^/]*$;-- &;' >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"


# --- File Contents ---
echo "File Contents:" >> "$OUTPUT_FILE"
echo "----------------" >> "$OUTPUT_FILE"
echo "(Snippets shown for input data files like *.txt)" >> "$OUTPUT_FILE"

FILES_TO_CAT=(
  "pyproject.toml"
  "README.md"
  "requirements.txt"
  "src/voxelflex/config/default_config.yaml"
  "src/voxelflex/config/config.py"
  "src/voxelflex/data/validators.py"
  "src/voxelflex/data/data_loader.py"
  "src/voxelflex/models/cnn_models.py"
  "src/voxelflex/utils/file_utils.py"
  "src/voxelflex/utils/logging_utils.py"
  "src/voxelflex/utils/system_utils.py"
  "src/voxelflex/utils/temp_scaling.py"
  "src/voxelflex/cli/cli.py"
  "src/voxelflex/cli/commands/preprocess.py"
  "src/voxelflex/cli/commands/train.py"
  "src/voxelflex/cli/commands/predict.py"
  "src/voxelflex/cli/commands/evaluate.py"
  "src/voxelflex/cli/commands/visualize.py"
  "input_data/train_domains.txt"
  "input_data/val_domains.txt"
  "input_data/test_domains.txt"
)

for file in "${FILES_TO_CAT[@]}"; do
  echo "" >> "$OUTPUT_FILE"
  echo "==========================================================" >> "$OUTPUT_FILE"
  if [ -f "$file" ]; then
      if [[ "$file" == input_data/*.txt ]]; then
          echo "===== FILE: $file (Top 10 Lines) =====" >> "$OUTPUT_FILE"
          echo "==========================================================" >> "$OUTPUT_FILE"
          echo "" >> "$OUTPUT_FILE"
          # Use head safely, suppressing error if file is empty/short
          head -n 10 "$file" 2>/dev/null >> "$OUTPUT_FILE"
          echo "" >> "$OUTPUT_FILE"
          echo "===== (End Snippet of $file) =====" >> "$OUTPUT_FILE"
      else
          echo "===== FILE: $file =====" >> "$OUTPUT_FILE"
          echo "==========================================================" >> "$OUTPUT_FILE"
          echo "" >> "$OUTPUT_FILE"
          # Use cat safely, suppressing error if file is unreadable (though unlikely if -f check passed)
          cat "$file" 2>/dev/null >> "$OUTPUT_FILE"
      fi
  else
    echo "===== FILE: $file (Not Found) =====" >> "$OUTPUT_FILE"
    echo "==========================================================" >> "$OUTPUT_FILE"
  fi
done

echo "" >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"
echo "              End of VoxelFlex Project Context            " >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"

echo "Finished generating $OUTPUT_FILE"
exit 0