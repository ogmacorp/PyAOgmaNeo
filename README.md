<!---
  PyAOgmaNeo
  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the PYAOGMANEO_LICENSE.md file included in this distribution.
--->

# PyAOgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction 

Welcome to the [Ogma](https://ogmacorp.com) PyAOgmaNeo library, which contains Python bindings to the [AOgmaNeo](https://github.com/ogmacorp/AOgmaNeo) library.

## Requirements

- OpenMP (this will likely already be installed on your system)
- pybind11 (will automatically install if not present)
- cmake

## Installation

You may install from pypi:

> pip install pyaogmaneo

Or from this directory:

> pip install .

This will download the AOgmaNeo library these bindings depend on automatically, and compile it.

Note that the branch of AOgmaNeo that will be used for building is based on the current branch of this repository (PyAOgmaNeo).
The build system will automatically download the AOgmaNeo branch of the same name as that currently checked out in this repository (using a specific commit id).

If you would like to use an existing system install of AOgmaNeo, set the following environment variable:

> export USE_SYSTEM_AOGMANEO

before installing.

## Importing and Setup

The PyAOgmaNeo module can be imported using:

```python
import pyaogmaneo
```

Refer to [the examples](./examples) for usage.
An API reference is available in the [docs directory](./docs). It can be built for easier viewing using [Sphinx](https://www.sphinx-doc.org/en/master/) (e.g. run `make html` in the docs directory).

## Contributions

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for information on making contributions to PyAOgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [PYAOGMANEO_LICENSE.md](./PYAOGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

PyAOgmaNeo Copyright (c) 2020-2022 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
