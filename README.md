# Molecular Graph Deep Sets
This repository contains an implementation of the molecular graph deep sets (MolSets) model for molecular mixture properties, associated with our paper [MolSets: Molecular graph deep sets learning for mixture property modeling](https://arxiv.org/abs/2312.16473).

![Model architecture](MolSets_architecture.webp)

## Citation
If you find this code useful, please consider citing the following paper:
```
@misc{zhang2023molsets,
      title={MolSets: Molecular graph deep sets learning for mixture property modeling}, 
      author={Hengrui Zhang and Jie Chen and James M. Rondinelli and Wei Chen},
      year={2023},
      eprint={2312.16473},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Requirements
MolSets requires the following packages:
- PyTorch >= 2.0
- PyG (`torch_geometric`)
- PyTorch Scatter (only for [DMPNN](https://github.com/itakigawa/pyg_chemprop))
The environment can be set up by running
```
conda env create -f environment.yml
```
But package compatibility issues may occur and need to be manually corrected. To run on GPUs, CUDA and GPU-enabled versions of PyTorch and PyG are requried.
