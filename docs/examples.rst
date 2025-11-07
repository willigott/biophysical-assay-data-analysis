Examples and Tutorials
======================

This section contains examples and tutorials for using the bada package.

DSF Analysis Tutorial
---------------------

For a comprehensive tutorial on DSF (Differential Scanning Fluorimetry) analysis, see our detailed notebook in the ``notebooks/`` directory:

* `DSF Analysis Tutorial <../notebooks/dsf_analysis.md>`_

This tutorial covers:

* Loading and parsing instrument data
* Data preprocessing and validation
* Feature extraction from melting curves
* Visualization of results
* Comparative analysis between samples

Example Data
------------

The package includes example data files for testing and learning:

* **QuantStudio7 example**: :download:`quantstudio7_example.csv <../notebooks/data/quantstudio7_example.csv>`

Basic Usage Examples
--------------------

Parsing Data
~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from bada.parsers import QuantStudio7Parser

    # Parse QuantStudio7 data
    data_path = Path("data/quantstudio7_example.csv")
    parser = QuantStudio7Parser(data_path)
    data = parser.parse()

    print(f"Loaded data with {len(data)} rows")
    print(f"Wells: {data['well_position'].unique()}")

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bada.processing import get_dsf_curve_features

    # Analyze a single well
    well_data = data[data["well_position"] == "A1"]
    features = get_dsf_curve_features(
        well_data, 
        min_temp=25.0, 
        max_temp=95.0, 
        smoothing=0.01
    )

    print(f"Melting temperature: {features['tm']:.2f}°C")
    print(f"Max derivative: {features['max_derivative_value']:.4f}")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

    from bada.visualization import create_melt_curve_plot

    # Create an interactive plot
    fig = create_melt_curve_plot(
        well_data,
        x_spline=features['x_spline'],
        y_spline=features['y_spline'], 
        y_spline_derivative=features['y_spline_derivative'],
        tm=features['tm']
    )
    fig.show()

Heatmap Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

    from bada.processing import get_dsf_curve_features_multiple_wells
    from bada.utils.reformatting import convert_features_to_plate_format
    from bada.visualization import create_heatmap_plot

    # Analyze all wells
    all_features = get_dsf_curve_features_multiple_wells(data)
    
    # Convert to plate format for heatmap
    plate_data, cols, rows = convert_features_to_plate_format(
        all_features, 
        plate_size=384,
        feature_name="tm"
    )
    
    # Create heatmap
    fig = create_heatmap_plot(
        plate_data, 
        cols, 
        rows, 
        title="Melting Temperatures (°C)",
        colorbar_title="Tm (°C)"
    )
    fig.show()

DTW Analysis
~~~~~~~~~~~~

.. code-block:: python

    from bada.processing import get_dtw_distances_from_reference
    from bada.utils.reformatting import convert_distances_to_plate_format

    # Calculate DTW distances from a reference well
    distances = get_dtw_distances_from_reference(
        data, 
        reference_well="A1", 
        normalized=True
    )
    
    # Convert to plate format
    plate_data, cols, rows = convert_distances_to_plate_format(
        distances, 
        plate_size=384
    )
    
    # Visualize distances
    fig = create_heatmap_plot(
        plate_data, 
        cols, 
        rows, 
        title="DTW Distances from A1",
        colorbar_title="Distance"
    )
    fig.show() 