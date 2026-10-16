# Proximity3D: Shape from Capacitive Proximity on Sensing Manifold

[Project Page](https://proximity3d.github.io/)

## Installation

Requirements: [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), [Git](https://git-scm.com/downloads), and an [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) compatible with CUDA 12.9.

```bash
git clone https://github.com/proximity3d/Proximity3D-Code.git
cd Proximity3D-Code
conda env create -f environment.yml
conda activate proximity3d
```

Install the additional packages in the activated environment following their official instructions. We have verified that these packages can be installed in the conda environment created above across multiple machines:

| Package | Repository |
| --- | --- |
| Kaolin | [NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin) |
| utils3d | [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d) |
| FlexGEMM | [JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) |
| CuMesh | [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh) |
| O-Voxel | [microsoft/TRELLIS.2/o-voxel](https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel) |
| nvdiffrast | [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast) |
| nvdiffrec renderutils | [JeffreyXiang/nvdiffrec](https://github.com/JeffreyXiang/nvdiffrec) |
| PyTorch3D | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) |

Then install this package:

```bash
python -m pip install -e .
```

The environment is now ready for the following steps.

## Dataset

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1T0l6BC602K7-7ESiUV8tXe_DzJbeayMq?usp=sharing).

After extracting the dataset, update the paths in `configs/data/datasets.json` to match your local data location.

## Training

After configuring the data paths, GPU count, batch size, and output directory, start training:

```bash
python -m proximity3d.train --config configs/proximity3d_384.toml
```

Checkpoints are saved in `<output_dir>/checkpoints/`: `last.ckpt` and the three best checkpoints ranked by validation loss. TensorBoard events are written under `<output_dir>`.

For the default output directory, monitor training with:

```bash
tensorboard --logdir LOGS/proximity3d
```

## Inference

Use a trained checkpoint and the matching model, sensor, and normalization configuration.

```bash
python -m proximity3d.infer \
  --checkpoint LOGS/proximity3d/checkpoints/last.ckpt
```

Reconstructed meshes are saved to `LOGS/inference`.

## ToDo

- [x] Release code
- [x] Release dataset
- [x] Release tutorial
