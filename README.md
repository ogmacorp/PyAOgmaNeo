<!---
  PyAOgmaNeo
  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the PYAOGMANEO_LICENSE.md file included in this distribution.
--->

# PyAOgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction 

Welcome to the [Ogma](https://ogmacorp.com) PyAOgmaNeo library, which contains Python bindings to the [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) library.

## Requirements

An install of [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) is required before installing the bindings. Make sure to build with `-DBUILD_SHARED_LIBS=On`.

Additionally this binding requires an installation of [SWIG](http://www.swig.org/) v3+

#### [SWIG](http://www.swig.org/)

- Linux requires SWIG installed via, for example ```sudo apt-get install swig3.0``` command (or via ```yum```).
- Windows requires installation of SWIG (v3). With the SourceForge Zip expanded, and the PATH environment variable updating to include the SWIG installation binary directory (for example `C:\Program Files (x86)\swigwin-3.0.8`).

## Installation

The following example can be used to build the Python package:

> python3 setup.py install --user  

or create a wheel file for installation via pip:

> python3 setup.py bdist_wheel  
> pip3 install dist/*.whl --user

Installation may fail if an old build directory still exists from a previous build. Make sure to remove such directories before rebuilding.

## Importing and Setup

The PyAOgmaNeo module can be imported using:

```python
import pyaogmaneo
```

Refer to the `WavyLinePrediction.py` example for further details.

## Contributions

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for information on making contributions to PyAOgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [PYAOGMANEO_LICENSE.md](./PYAOGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

PyAOgmaNeo Copyright (c) 2020 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
