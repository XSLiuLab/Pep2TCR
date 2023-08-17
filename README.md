# Pep2TCR: accurate prediction of CD4 T cell receptor binding specificity through transfer learning and ensemble approach

[![License](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/License-MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Pep2TCR can serve as a valuable tool for CD4 TCR specificity prediction and biology applications. This Github repository comprises the codes of Pep2TCR, providing comprehensive guidance for researchers interested in Pep2TCR.  Also, we provide a docker image at [Docker Hub](https://hub.docker.com/repository/docker/liuxslab/pep2tcr/general) and a website at [http://pep2tcr.liuxslab.com](http://pep2tcr.liuxslab.com) for convenient usage. 

## Contents

Users can download Pep2TCR package with `git clone https://github.com/XSLiuLab/Pep2TCR.git`, there are some contents:

- [data](./data) contains collected data used in the project.
- [model](./model) contains Pep2TCR model codes.
- [TCR_web](./TCR_web) contains Pep2TCR website codes.
- [Pep2TCR.py](./Pep2TCR.py) is a startup interface.
- [environment.yaml](./environment.yaml) is a user-friendly yaml file for creating a conda environment.

## Environment

We used Pytorch to train and validate Pep2TCR, so users should install the following packages:

- python == 3.8.13
- pytorch == 1.12.0 
- pandas == 1.5.3
- numpy == 1.23.5
- scikit-learn == 1.2.2

If the user's system is equipped with a GPU, they can install `cudatoolkit == 11.3.1`, which will result in an acceleration of prediction speed. 

Users also can setting up Conda environment through `conda env create -f environment.yaml`, this might take a bit of time, but it's incredibly convenient. In addition, please rewrite your `.condarc` file as following:

```bash
channels:
  - conda-forge
  - bioconda
  - menpo
  - main
  - r
  - msys2
  - pytorch
  - pytorch-lts
  - simpleitk
show_channel_urls: true
```

## Usage

Please modify the `ab_path` of the paras.py file in model\code folder to `/path/to/model_dir` as the first time use.

Pep2TCR has two modes: Single mode and Batch mode. The help page of Pep2TCR is as follows:

``` python
usage: Pep2TCR.py [-h] [--mode {single,batch}] [--cdr3 CDR3] [--pep PEP] [--data_path DATA_PATH] [--outdir OUTDIR]

optional arguments:
  -h, --help            show this help message and exit
  --mode {single,batch} default is single mode
  --cdr3 CDR3
  --pep PEP
  --data_path DATA_PATH csv format, first column is CDR3, second column is Epitope
  --outdir OUTDIR       default is . (current directory)
```

For Single mode, the command line is `python Pep2TCR.py --mode single --cdr3 xxx --pep xxx`

For Batch mode, the command line is `python Pep2TCR.py --mode batch --data_path /file path --outdir .`

## Citation

Waiting ... ...

## Acknowledgement

We acknowledge the computing services provided by ShanghaiTech University High Performance Computing Public Service Platform. This work received supports from the Shanghai Science and Technology Commission (21ZR1442400), the National Natural Science Foundation of China (31771373), and startup funding from ShanghaiTech University.

## License

***

**Cancer Biology Group @ShanghaiTech**

**Research group led by Xue-Song Liu in ShanghaiTech University**

