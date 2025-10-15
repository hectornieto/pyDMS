# Getting started

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

    If GDAL is installed in a non-standard location, you may need to set the ```GDAL_LIBRARY_PATH``` environment variable before installing.