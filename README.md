[![License: MIT](https://img.shields.io/badge/License-MIT-blueviolet)](https://github.com/akuhnregnier/bounded-rand-walkers/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Build Status](https://travis-ci.com/akuhnregnier/bounded-rand-walkers.svg?branch=master)](https://travis-ci.com/akuhnregnier/bounded-rand-walkers)
[![codecov](https://codecov.io/gh/akuhnregnier/bounded-rand-walkers/branch/master/graph/badge.svg)](https://codecov.io/gh/akuhnregnier/bounded-rand-walkers)

# bounded-rand-walkers

Generation of bounded random walks and analysis thereof.

Random walk data is generated using `C++` and analysed using `Python 3`.

# Usage

For detailed examples, please see [examples](https://github.com/akuhnregnier/bounded-rand-walkers/blob/master/examples).

## Data Generation

The `bounded_rand_walkers.cpp` sub-package contains compiled code for fast data generation, and a Python module meant for coordinating this process and reading (and binning) the resulting saved data files.
Within `bounded_rand_walkers.cpp`:
 - `generate_data`: Generate random walk data using one of a set of pre-defined pdfs in either 1D or 2D and a selection of pre-defined boundaries for 2D.
 - `get_binned_2D`: Iteratively load saved data files generated using `generate_data`, while aggregating the results using binning.

# Installation

### Python dependencies

For required `Python` packages (and both `xtensor` and `xtensor-python` which are required for the `C++` code) see `requirements.yaml`.
It is recommended to install the required packages into a new `conda` environment using:
```bash
conda env create --file requirements.yaml
```
Followed by
```bash
pip install ./bounded-rand-walkers
```
where `bounded-rand-walkers` refers to the directory containing this repository.

### C++ dependencies

For the `C++` code, the following dependencies are required:
 - [`xtensor`](https://xtensor.readthedocs.io/en/latest/installation.html) (installed via `conda`)
 - [`xtensor-python`](https://github.com/xtensor-stack/xtensor-python) (installed via `conda`)
 - [`nlopt`](https://github.com/stevengj/nlopt)
 - [`boost`](https://www.boost.org/)

On Linux, `nlopt` and `boost` can usually be installed using your package manager.
You may need to run `ldconfig` post-installation.

Using `Ubuntu`, for example, `nlopt` and `boost` can be installed using
```bash
sudo apt-get install libnlopt-cxx-dev libboost-all-dev
sudo ldconfig
```

### Jupyter notebook extensions:

Jupyter notebook extensions may be installed using the commands below.

```bash
jupyter nbextensions_configurator enable --user
```

JupyterLab ipywidgets:
```sh
conda install ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

To install Jupyterlab Code Formatter (more details at (https://jupyterlab-code-formatter.readthedocs.io/en/latest/installation.html):
 - `jupyter labextension install @ryantam626/jupyterlab_code_formatter`
 - `conda install -c conda-forge jupyterlab_code_formatter`
 - `jupyter serverextension enable --py jupyterlab_code_formatter`

## Installation (C++ code compilation)

**On Unix (Linux, OS X)**

 - clone this repository
 - `pip install ./bounded-random-walkers`

**On Windows (Requires Visual Studio 2015)**

 - For Python 3.5:
     - clone this repository
     - `pip install ./bounded-random-walkers`
 - For earlier versions of Python, including Python 2.7:

   xtensor requires a C++14 compliant compiler (i.e. Visual Studio 2015 on
   Windows). Running a regular `pip install` command will detect the version
   of the compiler used to build Python and attempt to build the extension
   with it. We must force the use of Visual Studio 2015.

     - clone this repository
     - `"%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64`
     - `set DISTUTILS_USE_SDK=1`
     - `set MSSdk=1`
     - `pip install ./bounded-random-walkers`

   Note that this requires the user building `bounded-random-walkers` to have registry edition
   rights on the machine, to be able to run the `vcvarsall.bat` script.


Windows runtime requirements
----------------------------

On Windows, the Visual C++ 2015 redistributable packages are a runtime
requirement for this project. It can be found [here](https://www.microsoft.com/en-us/download/details.aspx?id=48145).

If you use the Anaconda python distribution, you may require the Visual Studio
runtime as a platform-dependent runtime requirement for you package:

```yaml
requirements:
  build:
    - python
    - setuptools
    - pybind11

  run:
   - python
   - vs2015_runtime  # [win]
```
