# REV-U  
_Research EValuation Units for mapping interdisciplinary impact_

This repository provides an open-source implementation of REV-U, a clustering-based framework that enables domain-independent research evaluation by grouping scientific papers into interpretable units and quantifying distances between them.

The method combines natural language and authorship features via kernel methods and measures inter-cluster distances using the Wasserstein metric.
It supports exploratory, interdisciplinary analysis by visualizing research landscapes and identifying related work across fields—without relying on prior expert knowledge.

## Features
- [✔️] Clustering of scientific papers into evaluation units (clusters)
- [✔️] Visualization of interdisciplinary spillover
- [✔️] Dataset loader for S2AG and SPECTER features
- [✔️] Wasserstein distance-based similarity calculation

## Requirements

This project uses [Poetry](https://python-poetry.org/) for dependency and environment management.

Make sure you have Poetry installed. You can install it with:
## Installation
```bash
git clone https://github.com/ISM-REDi/player.git
cd player
poetry install
```


## How to Run

### Settings

This project uses two configuration files:

  bin/settings.conf: the main settings file

  src/config.py: Python-side logic for loading and interpreting configuration values

#### 1. Check or edit `setting.conf`

The setting.conf file contains key settings such as data paths, model parameters, and output destinations.
It typically looks like this:

[DEFAULT]
DATA_DIR="datas/evalunit/"
SAVE_DIR="results/evalunit/"

Edit this file to adjust parameters before execution.

#### 2. Write your API key in `config.py`

To use the S2AG (Semantic Scholar Academic Graph) API, you need to set your API key in the `config.py` file.  
For example:

```python
headers = {"x-api-key": "your-api-key-here"}
```

### Basic Execution

You can run the script with the following command:

```bash
bin/units.sh
```


## Project Structure

project-root/
├── pyproject.toml # Poetry configuration
├── bin/
│ ├── setting.conf # Project-wide configuration file
│ └── units.sh # Main script for batch execution
├── datas/ # Input datasets
├── results/ # Output and evaluation results
├── src/
│ ├── player/ # Core logic for clustering and evaluation
│ ├── s2ag/ # Modules for fetching and processing data from S2AG
│ ├── utils/ # Utility functions used across modules
│ ├── visual/ # Visualization scripts for evaluation units
│ ├── experiments/ # Sample scripts for experiments
│ └── config.py # Logic for loading settings (e.g., API key)
└── README.md # Project documentation


### Explanation

- `bin/setting.conf`: Stores configurable parameters such as data paths and the number of trials.
- `bin/units.sh`: Bash script for executing the evaluation pipeline.
- `src/config.py`: Loads configuration values, including the API key for S2AG.
- `src/player/`: Contains core scripts for clustering and evaluation.
- `src/s2ag/`: Handles API requests and data preprocessing using the Semantic Scholar Academic Graph.
- `src/utils/`: General-purpose utility functions.
- `src/visual/`: Scripts for visualizing the evaluation units and results.
- `src/experiments/`: Sample scripts for running reproducible experiments.
