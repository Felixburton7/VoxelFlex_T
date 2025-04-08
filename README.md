Okay, here is a detailed summary of the VoxelFlex project based on the context provided and our discussion.

**1. Project Aim & Goal**

*   **Scientific Goal:** To develop a deep learning model that accurately predicts per-residue protein flexibility, quantified by the Root Mean Square Fluctuation (RMSF), directly from 3D structural information.
*   **Key Innovation (Temperature Awareness):** A critical feature is the model's ability to incorporate temperature as an input feature. This allows a single trained model to predict RMSF values at *arbitrary* input temperatures, providing insights into protein dynamics across different thermal conditions.
*   **Technical Goal:** To create a robust and efficient Python package (`voxelflex`) using 3D Convolutional Neural Networks (CNNs). The input is a voxelized representation of the local atomic environment around each residue. A major technical challenge addressed is handling the extremely large size of potential processed voxel data for the entire dataset.

**2. Biological Significance**

*   Protein flexibility (RMSF) is fundamental to understanding protein function, including enzyme catalysis, molecular recognition (binding interactions), allostery, and overall stability.
*   Temperature significantly influences protein dynamics. A temperature-aware model allows researchers to study how flexibility changes with thermal conditions, crucial for understanding temperature adaptation in organisms (thermophiles vs. psychrophiles), optimizing enzymes for biotechnology at specific operating temperatures, and assessing protein stability under stress.

**3. Core Workflow/Strategy: "Metadata Preprocessing / On-Demand HDF5"**

This is the cornerstone of the project's data handling, designed specifically to manage large datasets efficiently without requiring massive intermediate storage:

*   **Challenge:** Storing fully processed voxel data (e.g., float32 tensors) for every residue at every temperature would likely require terabytes of storage, which is often impractical.
*   **Strategy:** Avoid storing processed voxels. Keep the raw voxel data in the source HDF5 and load/process it only when needed during training or prediction.
*   **`preprocess` Step (Metadata Only):**
    *   Reads the aggregated RMSF data (CSV).
    *   Reads the list of domain IDs present as keys in the HDF5 file.
    *   Reads the domain ID lists for train/validation/test splits (TXT files).
    *   **Maps** the RMSF data to the HDF5 structure, identifying which residues exist in HDF5 and have corresponding RMSF data at various temperatures.
    *   **Filters** based on the provided splits, assigning 'train', 'val', or 'test' to each valid residue@temperature data point.
    *   **Generates Output:**
        *   `master_samples.parquet`: A single file containing *only metadata* for every valid sample (e.g., `hdf5_domain_id`, `resid_str`, `raw_temp`, `target_rmsf`, `split`). **Crucially, no voxel data is stored here.** This file is relatively small.
        *   `temp_scaling_params.json`: Stores the min/max temperature found in the *training* split portion of the RMSF data, used for normalizing temperature input to the model.
        *   `failed_preprocess_domains.txt`: Lists domains that couldn't be processed (e.g., not found in HDF5, mapping issues).
*   **`train` Step (On-Demand Loading):**
    *   Uses the custom `ChunkedVoxelDataset` (a PyTorch `IterableDataset`).
    *   Each `DataLoader` worker (`num_workers`) reads the `master_samples.parquet` file to build an in-memory dictionary (`metadata_lookup`) mapping `(hdf5_domain_id, resid_str)` to its associated `(raw_temp, target_rmsf)` pairs for its assigned split ('train' or 'val').
    *   Workers iterate through their assigned list of domains in **chunks** (`chunk_size`).
    *   For each chunk, the worker reads the **raw voxel data** (e.g., boolean arrays) for the residues in that chunk **directly from the large HDF5 file** into its own RAM.
    *   The worker then processes these raw voxels (boolean to float32, transpose axes) **in its memory**.
    *   It uses the `metadata_lookup` to find the corresponding temperature(s) and target RMSF(s) for each processed residue voxel.
    *   It scales the temperature using the loaded `temp_scaling_params.json`.
    *   It creates the final sample dictionary containing PyTorch tensors for the processed voxel grid, the scaled temperature, and the target RMSF.
    *   These samples are `yield`ed to the main `DataLoader`.
    *   The worker **discards the raw voxel data** for the completed chunk from its RAM before reading the next chunk, managing memory usage.
    *   The main process collates yielded samples into batches (`batch_size`) and sends them to the GPU for training.
*   **`predict` / `evaluate` Steps:** Also use on-demand loading (`PredictionDataset`) to fetch necessary voxel data directly from HDF5 as needed for the specific domains/residues being predicted or evaluated.

**4. Input Data**

*   **Voxel Data (HDF5):**
    *   **Path:** Specified by `input.voxel_file` in config (e.g., `input_data/voxel/mdcath_voxelized.hdf5`).
    *   **Format:** HDF5 (`.hdf5`).
    *   **Structure:** Hierarchical. A typical path to a dataset looks like `HDF5_File[DomainID][ChainID][ResidueID]`.
        *   `DomainID`: Top-level key (e.g., `'1abcA00'`).
        *   `ChainID`: Group under DomainID (e.g., `'A'`).
        *   `ResidueID`: Key under ChainID, **string representation of the residue number** (e.g., `'123'`).
        *   *Dataset:* The actual voxel data associated with that residue.
    *   **Dataset Details:** Expected to be primarily `boolean` type, with shape `(21, 21, 21, 5)`. The 5 channels likely represent different atom types or properties within the 21x21x21 cube centered on the residue's alpha-carbon. This is processed on-the-fly to `float32` with shape `(5, 21, 21, 21)` (Channels-First format for PyTorch CNNs).
    *   **Example Access (Conceptual):** `h5file['1abcA00']['A']['123'][:]` would return the (21, 21, 21, 5) boolean numpy array for residue 123 of chain A in domain 1abcA00.
*   **Aggregated RMSF Data (CSV):**
    *   **Path:** Specified by `input.aggregated_rmsf_file` (e.g., `input_data/rmsf/aggregated_rmsf_all_temps.csv`).
    *   **Format:** Comma-Separated Values (`.csv`).
    *   **Content:** Contains RMSF values for residues across multiple domains and multiple temperatures.
    *   **Required Columns:** `domain_id` (matches/mappable to HDF5 DomainIDs), `resid` (integer residue number), `resname` (e.g., 'ALA'), `temperature_feature` (temperature in Kelvin, float), `target_rmsf` (the ground truth RMSF value, float).
    *   **Optional Columns:** `relative_accessibility`, `dssp`, `secondary_structure_encoded` (used for evaluation stratification).
    *   **Example Row:** `1abcA00,123,ALA,320.0,0.5512` (RMSF for Ala 123 in 1abcA00 at 320K is 0.5512). Note that residue 123 would have other rows for other temperatures (348K, 379K, etc.).
*   **Domain Split Files (.txt):**
    *   **Paths:** `input.train_split_file`, `input.val_split_file`, `input.test_split_file`.
    *   **Format:** Plain text (`.txt`).
    *   **Content:** Each file lists HDF5 `DomainID` keys, one per line, defining which domains belong to the training, validation, or test set. Ensures splits are done at the domain level to prevent data leakage between structurally similar proteins.
    *   **Example Line:** `1abcA00`

**5. Output Data**

Generated within the `outputs/<run_name>/` directory:

*   **From `preprocess`:**
    *   `input_data/processed/master_samples.parquet`: The crucial metadata file (domain, residue, temp, target RMSF, split).
    *   `outputs/<run_name>/models/temp_scaling_params.json`: Min/Max temperatures from training data.
    *   `outputs/<run_name>/failed_preprocess_domains.txt`: List of domains skipped during preprocessing.
*   **From `train`:**
    *   `outputs/<run_name>/models/*.pt`: Saved model checkpoints (`best_model.pt`, `latest_model.pt`, periodic checkpoints). Contain model weights, optimizer state, etc.
    *   `outputs/<run_name>/training_history.json`: Epoch-wise metrics (loss, pearson) and learning rate saved after training finishes.
    *   `outputs/<run_name>/logs/voxelflex.log`: Detailed log file for the run.
*   **From `predict`:**
    *   `outputs/<run_name>/metrics/predictions_*.csv`: CSV file containing predicted RMSF values for specified domains/residues at a target temperature.
*   **From `evaluate`:**
    *   `outputs/<run_name>/metrics/evaluation_metrics_*.json`: JSON file with calculated performance metrics (overall, stratified, permutation importance).
*   **From `visualize`:**
    *   `outputs/<run_name>/visualizations/*.png` (or other format): Generated performance plots (loss curves, scatter plots, error distributions, etc.).
    *   `outputs/<run_name>/visualizations/*_data.csv`: Optional CSV files containing the data used to generate each plot.

**6. Folder Structure**

(Snapshot from context, excluding transient/output folders)
```
.
├── input_data
│   ├── processed  # Location for master_samples.parquet
│   ├── rmsf       # Location for aggregated_rmsf_all_temps.csv
│   ├── test_domains.txt
│   ├── train_domains.txt
│   ├── val_domains.txt
│   └── voxel      # Location for mdcath_voxelized.hdf5
├── LICENSE
├── pyproject.toml # Project metadata, dependencies, entry points
├── README.md
├── requirements.txt
├── src
│   └── voxelflex  # Main package source code
│       ├── cli    # Command Line Interface logic
│       │   └── commands # Individual command implementations (train, predict...)
│       ├── config # Configuration loading and defaults
│       ├── data   # Data loading classes and validation
│       ├── models # CNN model definitions
│       └── utils  # Helper utilities (files, logging, system, etc.)
├── tests          # Unit/integration tests (currently placeholder)
└── # Other scripts like create_voxelflex_context.sh etc.
```

**7. Key Files and Their Roles**

*   **`src/voxelflex/cli/cli.py`:** Main entry point, parses arguments, sets up logging, loads config, calls command functions.
*   **`src/voxelflex/cli/commands/preprocess.py`:** Implements the metadata preprocessing logic.
*   **`src/voxelflex/cli/commands/train.py`:** Implements the training loop, validation, checkpointing, calls data loaders and model. Includes `train_epoch` and `validate` functions.
*   **`src/voxelflex/cli/commands/predict.py`:** Implements prediction using a trained model.
*   **`src/voxelflex/cli/commands/evaluate.py`:** Calculates metrics based on predictions and ground truth.
*   **`src/voxelflex/cli/commands/visualize.py`:** Generates plots from results.
*   **`src/voxelflex/config/default_config.yaml`:** Default parameters for the pipeline.
*   **`src/voxelflex/config/config.py`:** Loads user config, merges with defaults, validates, handles paths.
*   **`src/voxelflex/data/data_loader.py`:** Contains the crucial `ChunkedVoxelDataset` and `PredictionDataset` classes implementing the on-demand loading, plus helper functions like `load_aggregated_rmsf_data`.
*   **`src/voxelflex/data/validators.py`:** Functions to validate input data formats (primarily the RMSF CSV).
*   **`src/voxelflex/models/cnn_models.py`:** Defines the 3D CNN architectures (DenseNet3D, DilatedResNet3D, MultipathRMSFNet) adapted for voxel + temperature input.
*   **`src/voxelflex/utils/*.py`:** Various helper modules for file handling, advanced logging, system resource interaction, and temperature scaling logic.
*   **`pyproject.toml`:** Defines the package structure, dependencies, and the `voxelflex` command-line script entry point.

**8. Compute Environment Considerations**

*   The project runs on a Linux machine with substantial resources (36 CPU cores, ~63GB RAM, NVIDIA Quadro RTX 8000 GPU with ~48GB VRAM).
*   The GPU VRAM is crucial for fitting the 3D CNN model and data batches.
*   System RAM usage is significant due primarily to the `DataLoader` workers holding voxel chunks and the main process holding the metadata lookup (if not shared efficiently, though separate processes get copies). Tuning `num_workers` and `chunk_size` is key to managing RAM.
*   Disk I/O performance for the HDF5 file is critical for training speed, especially given the history of issues with the `/dev/sdc2` mount.

**9. Data Scale**

*   The dataset is large: ~5400 unique domains, >3.3 million residue@temperature points in the RMSF CSV, leading to >3.3 million samples in `master_samples.parquet`.
*   The raw HDF5 voxel file is likely hundreds of GBs.
*   This large scale necessitates the specific "Metadata Preprocessing / On-Demand HDF5" workflow.

This summary covers the project's purpose, how it handles data, its structure, and the rationale behind its design, incorporating details from the context file and our subsequent discussions.
