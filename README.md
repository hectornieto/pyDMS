[![PyPI version](https://img.shields.io/pypi/v/python_dms.svg?color=blue)](https://pypi.org/project/python_dms/)
[![Tests](https://github.com/radosuav/pyDMS/actions/workflows/tests.yml/badge.svg)](https://github.com/radosuav/pyDMS/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-purple)](https://radosuav.github.io/pyDMS/)

Read the documentation for more details: [https://radosuav.github.io/pyDMS/](https://radosuav.github.io/pyDMS/)

# pyDMS
Python implementation of Data Mining Sharpener (DMS): a decision tree based algorithm for
sharpening (disaggregation) of low-resolution images using high-resolution images.
The implementation is mostly based on [Gao2012].

The DMS is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.

The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees
    ([Gao2012], section 2.1), performing local (moving window) and global regressions and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual
    analysis and bias correction ([Gao2012], section 2.4)

Additionally, the Decision Tree regressor can be replaced by Neural Network regressor.

## Installation
```python_dms``` requires GDAL to be installed on your system.
Since GDAL provides compiled binaries that vary by platform, this package does not bundle them. You will need to install GDAL yourself before installing ```python_dms```.

1. **Install GDAL**
    #### macOS (Homebrew)
    ```
    brew install gdal
    ```
    #### Ubuntu / Debian
    ```
    sudo apt-get update
    sudo apt-get install -y gdal-bin libgdal-dev
    ```
    #### Windows
    Download and install GDAL from GIS Internals or install via conda:
    ```
    conda install -c conda-forge gdal
    ``` 

    You can verify your installation by running:
    ```
    gdalinfo --version
    ``` 

2. **Install ```python_dms```**

    Once GDAL is installed and available in your environment, install python_dms with pip:
    ```
    pip install python_dms
    ```

    or download the project to your local system, enter the download directory and then type

    `pip install .`

## Usage
For usage template see [run_pyDMS.py](/run_pyDMS.py).

Copyright: (C) 2024 Radoslaw Guzinski and contributors.

## References

* [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data
       Mining Approach for Sharpening Thermal Satellite Imagery over Land.
       Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287

* [Guzinski2019] Guzinski, R., & Nieto, H. (2019). Evaluating the feasibility of using Sentinel-2 and Sentinel-3 satellites for high-resolution evapotranspiration estimations. Remote Sensing of Environment, 221, 157–172. https://doi.org/10.1016/j.rse.2018.11.019

* [Guzinski2023] Guzinski, R., Nieto, H., Ramo Sánchez, R., Sánchez, J.M., Jomaa, I., Zitouna-Chebbi, R., Roupsard, O., and López-Urrea, R. (2023). Improving field-scale crop actual evapotranspiration monitoring with Sentinel-3, Sentinel-2, and Landsat data fusion. International Journal of Applied Earth Observation and Geoinformation 125, 103587. https://doi.org/10.1016/j.jag.2023.103587


## License

pyDMS: a Python Data Mining Sharpener implementation

Copyright 2024 Radoslaw Guzinski and contributors.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
