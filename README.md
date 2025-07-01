# TITLE  
_... research evaluation unit ~ ..._

This repository provides an open-source implementation of REV-U, a clustering-based framework that enables domain-independent research evaluation by grouping scientific papers into interpretable units and quantifying distances between them.

The method combines natural language and authorship features via kernel methods and measures inter-cluster distances using the Wasserstein metric.
It supports exploratory, interdisciplinary analysis by visualizing research landscapes and identifying related work across fields—without relying on prior expert knowledge.

## Features
- [✔️] Clustering of scientific papers into evaluation units (clusters)
- [✔️] Wasserstein distance-based similarity calculation
- [✔️] Visualization of interdisciplinary spillover
- [✔️] Dataset loader for S2AG and SPECTER features

## Installation

```bash
git clone https://github.com/ISM-REDi/player.git
cd yourproject
poetry install
```

## Usage
```bash
bin/start.sh
# poetry run python ...
```

