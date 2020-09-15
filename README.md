# bounded-rand-walkers

Generation of bounded random walks and analysis thereof.

Random walk data is generated using `C++` and analysed using `Python 3`.

# Installation

For required `Python` packages (and `xtensor`, which is required for the `C++` code) see `requirements.txt`.

For the `C++` code, the following dependencies are required:
 - [`xtensor`](https://xtensor.readthedocs.io/en/latest/installation.html) (may be installed via conda)
 - [`xtensor-python`](https://github.com/xtensor-stack/xtensor-python) (may be installed via conda)
 - [`nlopt`](https://github.com/stevengj/nlopt)
 - [`boost`](https://www.boost.org/)

You might have to run `ldconfig` post-installation on Linux.

Installation (C++ code compilation)
-----------------------------------

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
