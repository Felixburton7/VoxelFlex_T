# VoxelFlex (Temperature-Aware - Preprocessing Workflow)

Predicting temperature-dependent protein flexibility (RMSF) from 3D voxel data using deep learning, optimized with a robust preprocessing pipeline.

## Overview

This package provides tools to train and use 3D Convolutional Neural Networks (CNNs) for predicting per-residue Root Mean Square Fluctuation (RMSF) values. This version is **temperature-aware**, meaning it takes the simulation temperature as an input feature, allowing a single model to predict flexibility across different temperatures.

It utilizes voxelized representations of protein structures (e.g., from Aposteriori) and RMSF data derived from Molecular Dynamics (MD) simulations (e.g., from the mdCATH dataset).

**Key Feature:** This version implements an optimized **preprocessing workflow**. Raw voxel and RMSF data are converted into batched PyTorch tensor files (`.pt`) before training. This significantly simplifies the training loop, improves performance, enhances robustness, and allows for efficient handling of very large datasets within defined memory limits.

## Features

*   Temperature-aware 3D CNN models (MultipathRMSFNet, DenseNet3D, DilatedResNet3D).
*   **Robust preprocessing pipeline:** Converts raw HDF5/CSV to optimized `.pt` batch files.
    *   Handles HDF5 boolean dtype casting and shape transposition.
    *   Manages memory via batch-wise HDF5 loading and optional caching.
    *   Scales temperature feature based on training set statistics.
    *   Generates metadata files (`.meta`) for easy loading.
*   **Simplified and efficient training:** Uses preprocessed batches directly in the DataLoader.
*   Support for mixed-precision training and standard optimizers/schedulers.
*   Rigorous evaluation including stratified metrics and permutation feature importance for temperature.
*   Visualization tools for analyzing training progress and model performance.
*   Command-line interface (`preprocess`, `train`, `predict`, `evaluate`, `visualize`).
*   Designed for use with large-scale MD datasets like mdCATH.

## Installation

```bash
# Recommended: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows

# Install the package from the project root directory
pip install .

# Or for development (changes in src/ reflect immediately):
pip install -e .
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The package is primarily used via the command line interface.

**1. Preprocess Data:** (Run this first!)
```bash
voxelflex preprocess --config path/to/your/config.yaml [-v|-vv]
```
This reads raw data specified in the config, performs processing and scaling, and saves batches to `input_data/processed/` (or as configured in `data.processed_dir`).

**2. Train Model:**
```bash
voxelflex train --config path/to/your/config.yaml [-v|-vv] [--force_preprocess]
```
This loads the preprocessed data and trains the model, saving outputs (checkpoints, logs) to `outputs/<run_name>/`. Use `--force_preprocess` to re-run preprocessing before training.

**3. Predict RMSF:**
```bash
voxelflex predict --config path/to/config.yaml --model path/to/model.pt --temperature 320 [-v|-vv] [--domains ID1 ID2 ...] [--output_csv filename.csv]
```
Loads the trained model and predicts RMSF for specific domains (or test split domains) at the given temperature. Saves results to `outputs/<run_name>/metrics/`.

**4. Evaluate Model:**
```bash
voxelflex evaluate --config path/to/config.yaml --model path/to/model.pt --predictions path/to/preds.csv [-v|-vv]
```
Compares predictions against ground truth (from the aggregated RMSF file) and calculates performance metrics, saving results to `outputs/<run_name>/metrics/`.

**5. Visualize Results:**
```bash
voxelflex visualize --config path/to/config.yaml --predictions path/to/preds.csv [-v|-vv] [--history path/to/history.json]
```
Generates performance plots based on prediction/evaluation data and optional training history, saving them to `outputs/<run_name>/visualizations/`.

Use `-v` for INFO level logging and `-vv` for DEBUG level logging for any command.

## Configuration

Modify the `src/voxelflex/config/default_config.yaml` file or create a copy and adjust parameters. Key sections:

*   `input`: Paths to raw voxel HDF5, aggregated RMSF CSV, and domain split files (`.txt`).
*   `data`: Paths for processed data output, preprocessing batch size, cache limit.
*   `output`: Base directory for run outputs (logs, models, metrics, visualizations).
*   `model`: CNN architecture choice and hyperparameters.
*   `training`: Epochs, batch size (for loading `.pt` files), learning rate, optimizer, scheduler, etc.
*   `prediction`: Batch size for inference.
*   `evaluation`: Settings for stratified metrics and permutation importance.
*   `logging`: Logging levels and progress bar visibility.
*   `visualization`: Toggles for different plots and output settings.
*   `system_utilization`: GPU preference.

## Data Preparation

1.  **Voxel Data (HDF5):** Place your HDF5 file (e.g., `mdcath_voxelized.hdf5`) in `input_data/voxel/`. Ensure it follows the expected structure (`DomainID -> ChainID -> ResidueID -> Dataset`) and that datasets are predominantly `bool` type with shape `(X, Y, Z, Channels)` where `Channels=5`.
2.  **RMSF Data (CSV):** Create an aggregated CSV file (e.g., `aggregated_rmsf_all_temps.csv`) containing RMSF data from all relevant temperatures. Required columns: `domain_id` (must be mappable to HDF5 keys), `resid` (integer), `resname` (string), `temperature_feature` (float, Kelvin), `target_rmsf` (float). Optional columns for evaluation: `relative_accessibility`, `dssp` (or `secondary_structure_encoded`). Place in `input_data/rmsf/`.
3.  **Splits (.txt):** Generate plain text files listing HDF5 domain keys for training, validation, and testing (one ID per line). Place these in `input_data/`. Using splits generated by tools like `mdcath-sampling` is highly recommended to mitigate homology bias.

graph TD
    Start[Start Training Run (voxelflex train)] --> LoadConfig{Load Config (.yaml)};
    LoadConfig --> ReadParams[Read Params: num_workers, chunk_size, batch_size, etc.];
    LoadConfig --> CheckMetadata{Check for master_samples.parquet};

    subgraph Preprocessing [Optional: If Metadata Missing]
        direction LR
        P_Start(Start Preprocess) --> P_ReadRMSF(Read RMSF CSV)
        P_ReadRMSF --> P_ReadHDF5Keys(Read HDF5 Keys)
        P_ReadHDF5Keys --> P_Map(Map Domains/Residues)
        P_Map --> P_GenerateSamples(Generate Sample List)
        P_GenerateSamples --> P_WriteParquet(Write master_samples.parquet)
        P_WriteParquet --> P_End(End Preprocess)
    end

    CheckMetadata -- Found --> InitLoaders[Initialize DataLoaders];
    CheckMetadata -- Not Found --> Preprocessing --> InitLoaders;

    InitLoaders --> SpawnWorkers[Spawn DataLoader Workers (num_workers)];

    subgraph WorkerProcess [Worker Process (Repeated for num_workers)]
        direction TB
        W_Start(Worker Start) --> W_LoadMeta[Load master_samples.parquet];
        W_LoadMeta --> W_MetaLookup(Store Metadata Lookup in Worker RAM);
        W_MetaLookup --> W_GetDomains(Get Assigned Domain List);
        W_GetDomains --> W_LoopChunks{Loop Through Domain Chunks};

        W_LoopChunks -- Next Chunk --> W_ReadHDF5["Read Raw Voxels for 'chunk_size' Domains (from HDF5 on Disk)"];
        W_ReadHDF5 --> W_VoxelsRAM(Store Raw Voxels in Worker RAM);
        W_VoxelsRAM --> W_PrepSamples["Prepare Samples (Process Voxels, Scale Temp, Create Tensors in Worker RAM) using Metadata Lookup"];
        W_PrepSamples --> W_Yield(Yield Prepared Sample [Tensor Dict]);
        W_Yield --> W_ClearVoxels(Delete Raw Voxels for completed chunk from Worker RAM);
        W_ClearVoxels --> W_LoopChunks;
        W_Yield --> Q_Queue([Main Process DataLoader Queue]);

        %% Annotations
        style W_ReadHDF5 fill:#f9d,stroke:#333,stroke-width:2px;
        note right of W_ReadHDF5 Key Parameter: `chunk_size` <br/> Controls Worker RAM & HDF5 Read Size
    end

    SpawnWorkers --> WorkerProcess;


    subgraph MainProcess [Main Training Process]
        direction TB
        Q_Queue --> M_Collate["Collate 'batch_size' Samples from Queue"];
        M_Collate --> M_BatchRAM(Create Batch Tensors in Main Process RAM);
        M_BatchRAM --> M_Loop{Training Loop (Iterate Batches)};

        M_Loop -- Next Batch --> M_ToGPU["Move Batch Tensors to GPU VRAM"];
        M_ToGPU --> M_Model["Model Forward/Backward (on GPU)"];
        M_Model --> M_Optimize["Optimizer Step (Update Weights on GPU)"];
        M_Optimize --> M_Loop;

        %% Annotations
        style M_Collate fill:#ccf,stroke:#333,stroke-width:2px;
        note right of M_Collate Key Parameter: `batch_size` <br/> Controls GPU VRAM & Training Step Size

        M_Loop -- Epoch End --> M_Validate{Run Validation};
        M_Validate --> M_Checkpoint[Save Checkpoint (.pt to Disk)];
        M_Checkpoint --> M_History[Update History];
        M_History --> M_NextEpoch{Start Next Epoch?};
        M_NextEpoch -- Yes --> M_Loop;
        M_NextEpoch -- No --> M_End[End Training];
    end

    Start --> MainProcess;


## Contributing

[Optional: Add guidelines for contributions if applicable]

## License

[Specify License, e.g., MIT License] - Remember to add a LICENSE file.
# VoxelFlex_T
# VoxelFlex_T
