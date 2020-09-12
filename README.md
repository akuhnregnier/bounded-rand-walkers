# bounded-rand-walkers

Generation of bounded random walks and analysis thereof.

Random walk data is generated using `C++` and analysed using `Python 3`.

# Installation

For required `Python` packages (and `xtensor`, which is required for the `C++` code) see `requirements.txt`.

For the `C++` code, the [`xtensor`](https://xtensor.readthedocs.io/en/latest/installation.html), [`nlopt`](https://github.com/stevengj/nlopt), and [`cnpy`](https://github.com/rogersce/cnpy) libraries are required.

Compiling the `C++` code (the primary `data_generation` executable) can be done by running `make` in the directory `cpp/build/`:

```bash
cd cpp/build
make
```
