bada: Biophysical Assay Data Analysis
=====================================

**bada** is a Python package for analysis of biophysical assays, particularly Differential Scanning Fluorimetry (DSF) data.

The package provides parsers for different qPCR instruments, data processing capabilities, feature extraction, and visualization tools for analyzing thermal melting curves.

Features
--------

* **Multi-instrument support**: Parsers for QuantStudio7 and LightCycler480 instruments
* **Data validation**: Robust input validation using Pandera schemas
* **Feature extraction**: Automated extraction of melting temperatures and curve characteristics
* **Visualization**: Interactive plotting with Plotly for melting curves and heatmaps
* **DTW analysis**: Dynamic Time Warping for comparing melting curves
* **Flexible preprocessing**: Spline fitting and normalization utilities

Quick Start
-----------

.. code-block:: python

    from bada.parsers import QuantStudio7Parser
    from bada.processing import get_dsf_curve_features
    from bada.visualization import create_melt_curve_plot

    # Parse instrument data
    parser = QuantStudio7Parser("data.csv")
    data = parser.parse()

    # Extract features from a single well
    well_data = data[data["well_position"] == "A1"]
    features = get_dsf_curve_features(well_data)

    # Create visualization
    fig = create_melt_curve_plot(well_data)
    fig.show()

Installation
------------

Install bada using pip:

.. code-block:: bash

    pip install bada

Or for development:

.. code-block:: bash

    git clone https://github.com/willigott/biophysical-assay-data-analysis.git
    cd biophysical-assay-data-analysis
    uv sync --group dev

Contents
--------

.. toctree::
   :maxdepth: 2

   api
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`