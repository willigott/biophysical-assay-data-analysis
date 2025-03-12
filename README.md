# Biophysical Assay Data Analysis

A Python package for analyzing and visualizing Differential Scanning Fluorimetry (DSF) data. 

## Features

- data import from commonly biophysical instrument formats. Currently supported instruments are
  QuantStudio7 and Lightcycler 480.
- quick visualization for raw data and the signals first derivative
- convenient calculation of relevant features such as the melting temperature Tm (here: max first
  derivative)
- identification of atypical shapes of signals (e.g. empty wells, failed experiment)


## Installation

### From PyPI (Recommended)

The easiest way to install is via pip:

```bash
pip install bada
```

### From Source

To install the latest development version:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/biophysical-assay-data-analysis.git
   cd biophysical-assay-data-analysis
   ```

2. Create a virtual environment (optional but recommended) using e.g. uv

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

To get started with this package, please refer to the example Jupyter notebooks in the `notebooks`
folder. These notebooks provide step-by-step guides for common workflows and demonstrate the key
functionality of the package.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
