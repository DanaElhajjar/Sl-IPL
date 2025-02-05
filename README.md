# Sl-IPL

This repository is dedicated to the implementation of the Sliding Interferometric Phase Linking (IPL).

The repository provides reproduction of the results presented in the paper:
> Dana EL HAJJAR, Arnaud Breloy, Guillaume GINOLHAC, Mohammed Nabil EL KORSO and Yajing YAN, "Sliding IPL: an efficient approach for estimating the phases of large SAR image time series"

If you use any of the code or data provided here, please cite the above paper.

## Code organisation

├── environment.yml<br>
├── real_data<br>
│   ├── real_data.py<br>
├── README.md<br>
└── src<br>
    ├── estimation.py<br>
    ├── __init__.py<br>
    ├── optimization.py<br>

The main code for the methods is provided in src/ directory. The file optimization.py provides the function for the MM algorithm. The folder real_data/ provides the real data experiments. The data/ directory is used to store the dataset used.


## Environment

A conda environment is provided in the file `environment.yml` To create and use it run:

```console
conda env create -f environment.yml
conda activate sl-ipl
```


### Authors

* Dana El Hajjar, mail: dana.el-hajjar@univ-smb.fr, dana.el-hajjar@centralesupelec.fr
* Arnaud Breloy, mail: arnaud.breloy@lecnam.net
* Guillaume Ginolhac, mail: guillaume.ginolhac@univ-smb.fr
* Mohammed Nabil El Korso, mail: mohammed.nabil.el-korso@centralesupelec.fr
* Yajing Yan, mail: yajing.yan@univ-smb.fr

Copyright @Université Savoie Mont Blanc, 2025