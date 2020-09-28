# -*- coding: utf-8 -*-
import os
import sys

import setuptools
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension, setup

# Used to derive xtensor-python BuildExt class.
# from setuptools.command.build_ext import build_ext


with open("README.md", "r") as f:
    readme = f.read()


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


class get_numpy_include(object):
    """Helper class to determine the numpy include path

    The purpose of this class is to postpone importing numpy
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self):
        pass

    def __str__(self):
        import numpy as np

        return np.get_include()


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++14 compiler flag  and errors when the flag is
    no available.
    """
    if has_flag(compiler, "-std=c++17"):
        return "-std=c++17"
    else:
        raise RuntimeError("C++17 support required!")


compile_args = ["-O3", "-std=c++17"]

# ext_modules = []
ext_modules = [
    Extension(
        "_bounded_rand_walkers_cpp",
        ["src/bounded_rand_walkers/cpp/main.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            get_numpy_include(),
            os.path.join(sys.prefix, "include"),
            os.path.join(sys.prefix, "Library", "include"),
        ],
        library_dirs=["/usr/local/lib"],
        libraries=["nlopt", "m"],
        language="c++",
    )
]

include_dirs = [str(get_numpy_include())]
language = "C++"

ext_modules += cythonize(
    [
        Extension(
            "bounded_rand_walkers.c_g2D",
            sources=["src/bounded_rand_walkers/c_g2D.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=compile_args,
            language=language,
        ),
    ]
)


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.append("-O3")
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name="bounded_rand_walkers",
    author="Alexander Kuhn-Regnier",
    author_email="ahk114@ic.ac.uk",
    url="https://github.com/akuhnregnier/bounded-rand-walkers",
    description="Bounded random walker simulation.",
    long_description=readme,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.0.1", "numpy"],
    cmdclass={"build_ext": build_ext},
    # xtensor-python BuildExt class.
    # cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    setup_requires=["setuptools-scm"],
    use_scm_version=dict(write_to="src/bounded_rand_walkers/_version.py"),
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
