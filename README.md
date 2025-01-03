# SCANS

SCANS is designed for the accurate prediction of protein carbonylation sites.

The source code is available at https://github.com/jianzhang-xynu/SCANS

# Benchmark datasets

The benchmark_datasets folder contains sample lists and corresponding protein segments.

## The format of these segments include:

* A fixed length of 27 residues

* Carbonylation annotation: "1" indicates a carbonylation site, while "0" indicates a non-carbonylation site.

* Ligand interaction annotation: "1" indicates a ligand interaction site, while "0" indicates a non-ligand interaction site.

* Non-functional residue annotation: "1" indicates a non-functional residue, while "0" indicates a functional residue.

## Examples:

* RQLCELLKYAILGKSTLPKPSWCQLLH,1,0,0: This is a K-centered carbonylation segment, with the center K being a carbonylation site.

* DIDRDALYVTNAVKHFKFTRAAGGKRR,0,1,0: This is a K-centered ligand interaction segment, with the center K being a ligand interaction site.

* SRTRPADWYESLMKAYVIDTVSADFYR,0,0,1: This is a K-centered non-functional segment, with the center K being a non-functional residue.


# Computed features

The computed_features folder stores PSSM-based features and physicochemical properties, organized into the PSSMfeas and PCfeas folders, respectively.


# Selective carbonylation-related motifs

The selective_computation_based_motifs folder contains computed carbonylation-related motifs based on information theory.

# Source code installation 
## Required Libraries

```
pip install numpy
pip install scipy
pip install pytorch==1.12.0 (or GPU supported pytorch, refer to https://pytorch.org/ for instructions)
```

## Facebook's ESM2 model

For installation of ESM, visit https://github.com/facebookresearch/esm. You can use the following command for the latest release:

```
pip install fair-esm  # latest release, OR:
```

Or, for the bleeding edge (current repo main branch):

```
pip install git+https://github.com/facebookresearch/esm.git 
```

In this study, we utilize esm2_t6_8M_UR50D with 8M parameters to compute residue embeddings.

# Prediction

To run predictions, use the command:

```
python3 runPredictions.py
```


# Running on GPU or CPU

For GPU usage, ensure you have CUDA and cuDNN installed; refer to their respective websites for instructions. The code has been tested on both GPU and CPU-only systems.


# Citation

Upon the usage the users are requested to use the following citation:

Users are requested to cite the following work upon usage:

Jian Zhang, Jingjing Qian, Pei Wang, Xuan Liu, Fuhao Zhang, Haiting Chai, Quan Zou. Explainable deep multi-level attention learning modeling for protein carbonylation site prediction.
