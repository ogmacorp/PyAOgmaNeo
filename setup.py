# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# For developers, set to use system install of AOgmaNeo
use_system_aogmaneo = True if "USE_SYSTEM_AOGMANEO" in os.environ else False

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[
            "source/pyaogmaneo/PyHelpers.h",
            "source/pyaogmaneo/PyHelpers.cpp",
            "source/pyaogmaneo/PyHierarchy.h",
            "source/pyaogmaneo/PyHierarchy.cpp",
            "source/pyaogmaneo/PyImageEncoder.h",
            "source/pyaogmaneo/PyImageEncoder.cpp",
            "source/pyaogmaneo/PyModule.cpp",
            ])

        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))

            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [ '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                       '-DUSE_SYSTEM_AOGMANEO=' + ('On' if use_system_aogmaneo else 'Off') ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = [ '--config', cfg ]

        if platform.system() == "Windows":
            cmake_args += [ '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir) ]

            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']

            build_args += ['--', '/m']
        else:
            cmake_args += [ '-DCMAKE_BUILD_TYPE=' + cfg ]
            build_args += [ '--', '-j2' ]

        env = os.environ.copy()

        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call([ 'cmake', ext.sourcedir ] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call([ 'cmake', '--build', '.' ] + build_args, cwd=self.build_temp)

setup(
    name="pyaogmaneo",
    version="1.11.0",
    description="Python bindings for the AOgmaNeo library",
    long_description='https://github.com/ogmacorp/PyAOgmaNeo',
    author='Ogma Intelligent Systems Corp',
    author_email='info@ogmacorp.com',
    url='https://ogmacorp.com/',
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    ext_modules=[ CMakeExtension("pyaogmaneo") ],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    zip_safe=False,
    include_package_data=True,
)
