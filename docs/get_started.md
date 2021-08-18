# Prerequisites

* Windows, Linux or MacOS
* Python 3+
* Tensorflow 2+

# Installation

## Prepare environment

1. The very first thing to install *easyrec* is to ensure that Python is installed. If not, it is recommended to
   use [Anaconda](https://www.anaconda.com/) instead of pure Python.
2. Create a conda virtual environment.

```bash
conda create -n easyrec python=3.7
```

3. Activate the virtual environment.

```bash
conda activate easyrec
```

## Install *easyrec*

### PyPI

The easiest way to install *easyrec* is PyPI. You can directly run the following command to install:

```bash
pip install easyrec-python
```

### Source

In addition, you can also install *easyrec* from source. First, locate the directory that you want to keep codes:

```bash
cd /path/for/codes/
```

Next, clone the source codes from Github:

```bash
git clone git@github.com:xu-zhiwei/easyrec.git
```

Finally, install *easyrec* in your Python environment:

```bash
cd easyrec
pip install requirements.txt
python setup.py install # or "python setup.py develop" for developers who wants to modify the codes
```

## Verification

After installation, you can verify it by the following:

1. Switch to Python environment.

```bash
python
```

2. Verify installation.

```python
import easyrec
```

The above code is supposed to run successfully upon you finish installation.
