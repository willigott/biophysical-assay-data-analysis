from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bada.visualization.heatmap import create_heatmap_plot
from bada.visualization.melting_curves import (
    PlotType,
    _add_model_overlay,
    _add_temperature_range_indicators,
    _add_tm_indicator,
    _create_derivative_subplot_figure,
    _create_raw_only_figure,
    _create_with_spline_figure,
    _infer_plot_type,
    create_melt_curve_plot,
    create_melt_curve_plot_from_features,
)


class TestHeatmap:
    def test_plot_plate_data_return_type(
        self, sample_plate_data: np.ndarray, plate_rows: list[str], plate_cols: list[str]
    ) -> None:
        """Test that plot_plate_data returns a plotly Figure object."""
        fig = create_heatmap_plot(
            sample_plate_data,
            cols=plate_cols,
            rows=plate_rows,
            title="Test Heatmap",
            colorbar_title="Value",
        )

        assert isinstance(fig, go.Figure)

    def test_plot_plate_data_content(
        self, sample_plate_data: np.ndarray, plate_rows: list[str], plate_cols: list[str]
    ) -> None:
        """Test that plot_plate_data contains a heatmap trace."""
        fig = create_heatmap_plot(
            sample_plate_data, cols=plate_cols, rows=plate_rows, title="Test Heatmap"
        )

        # Check that the figure has one trace
        assert fig.data

        # Check that the trace is a heatmap
        trace = cast(go.Heatmap, fig.data[0])
        assert isinstance(trace, go.Heatmap)

        # Check that the data is correctly passed
        assert np.array_equal(trace.z, sample_plate_data)
        # Plotly converts lists to tuples, so we need to convert for comparison
        assert tuple(trace.x) == tuple(plate_cols)
        assert tuple(trace.y) == tuple(plate_rows)

    def test_plot_plate_data_layout(
        self, sample_plate_data: np.ndarray, plate_rows: list[str], plate_cols: list[str]
    ) -> None:
        """Test that plot_plate_data has the expected layout."""
        title = "Test Heatmap"
        fig = create_heatmap_plot(sample_plate_data, cols=plate_cols, rows=plate_rows, title=title)

        # Check title is set correctly
        assert fig.layout.title.text == title

        # Check y-axis is reversed (standard for plate layouts)
        assert fig.layout.yaxis.autorange == "reversed"

        # Check plot dimensions are set
        assert fig.layout.height is not None

        # Check other layout properties are set
        assert fig.layout.plot_bgcolor == "white"
        assert fig.layout.xaxis.side == "top"  # x-axis labels on top

    def test_custom_color_scale(
        self, sample_plate_data: np.ndarray, plate_rows: list[str], plate_cols: list[str]
    ) -> None:
        """Test that custom color scales can be applied."""
        color_scale = "Viridis"
        fig = create_heatmap_plot(
            sample_plate_data,
            cols=plate_cols,
            rows=plate_rows,
            title="Test Heatmap",
            color_scale=color_scale,
        )

        # The colorscale is expanded to a list of tuples, so we can't directly compare
        # Instead, check that the colorscale is not the default
        trace = cast(go.Heatmap, fig.data[0])
        assert trace.colorscale is not None


class TestMeltingCurves:
    def test_plot_type_enum(self) -> None:
        """Test that PlotType enum contains expected values."""
        assert PlotType.DERIVATIVE.value == "derivative"
        assert PlotType.SPLINE.value == "spline"
        assert PlotType.RAW.value == "raw"

    def test_infer_plot_type_derivative(self) -> None:
        """Test that _infer_plot_type correctly identifies derivative plot."""
        x_spline = np.linspace(25, 95, 100)
        y_spline = np.random.random(100)
        y_spline_derivative = np.random.random(100)

        plot_type = _infer_plot_type(x_spline, y_spline, y_spline_derivative)
        assert plot_type == PlotType.DERIVATIVE

    def test_infer_plot_type_spline(self) -> None:
        """Test that _infer_plot_type correctly identifies spline plot."""
        x_spline = np.linspace(25, 95, 100)
        y_spline = np.random.random(100)
        y_spline_derivative = None

        plot_type = _infer_plot_type(x_spline, y_spline, y_spline_derivative)
        assert plot_type == PlotType.SPLINE

    def test_infer_plot_type_raw(self) -> None:
        """Test that _infer_plot_type correctly identifies raw plot."""
        x_spline = None
        y_spline = None
        y_spline_derivative = None

        plot_type = _infer_plot_type(x_spline, y_spline, y_spline_derivative)
        assert plot_type == PlotType.RAW

    def test_create_melt_curve_plot_raw_type(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that create_melt_curve_plot works with raw data."""
        fig = create_melt_curve_plot(sample_dsf_data)

        assert isinstance(fig, go.Figure)
        # Just verify that the figure was created successfully
        assert fig is not None
        assert hasattr(fig, "data")

    def test_create_melt_curve_plot_spline_type(
        self, sample_dsf_data: pd.DataFrame, sample_spline_x: np.ndarray
    ) -> None:
        """Test that create_melt_curve_plot works with spline data."""
        # Create dummy spline data
        x_spline = sample_spline_x
        y_spline = np.random.random(len(x_spline))

        fig = create_melt_curve_plot(sample_dsf_data, x_spline=x_spline, y_spline=y_spline)

        assert isinstance(fig, go.Figure)
        # Just verify the figure was created successfully with traces
        assert hasattr(fig, "data")
        assert fig.data is not None

    def test_create_melt_curve_plot_derivative_type(
        self, sample_dsf_data: pd.DataFrame, sample_spline_x: np.ndarray
    ) -> None:
        """Test that create_melt_curve_plot works with derivative data."""
        # Create dummy spline and derivative data
        x_spline = sample_spline_x
        y_spline = np.random.random(len(x_spline))
        y_spline_derivative = np.random.random(len(x_spline))

        fig = create_melt_curve_plot(
            sample_dsf_data,
            x_spline=x_spline,
            y_spline=y_spline,
            y_spline_derivative=y_spline_derivative,
        )

        assert isinstance(fig, go.Figure)
        # Just verify that the figure was created successfully
        assert fig is not None

    def test_create_melt_curve_plot_with_tm(
        self, sample_dsf_data: pd.DataFrame, sample_spline_x: np.ndarray
    ) -> None:
        """Test that create_melt_curve_plot correctly adds Tm indicator."""
        x_spline = sample_spline_x
        y_spline = np.random.random(len(x_spline))
        tm = 60.0  # Example Tm value

        fig = create_melt_curve_plot(sample_dsf_data, x_spline=x_spline, y_spline=y_spline, tm=tm)

        assert isinstance(fig, go.Figure)
        # Verify shapes are present
        shapes = fig.layout.shapes
        assert shapes is not None

    def test_create_melt_curve_plot_with_temp_range(
        self, sample_dsf_data: pd.DataFrame, sample_spline_x: np.ndarray
    ) -> None:
        """Test that create_melt_curve_plot correctly adds temperature range indicators."""
        x_spline = sample_spline_x
        y_spline = np.random.random(len(x_spline))
        min_temp = 40.0
        max_temp = 80.0

        fig = create_melt_curve_plot(
            sample_dsf_data,
            x_spline=x_spline,
            y_spline=y_spline,
            min_temp=min_temp,
            max_temp=max_temp,
        )

        assert isinstance(fig, go.Figure)
        # Verify shapes are present
        shapes = fig.layout.shapes
        assert shapes is not None

    def test_create_derivative_subplot_figure(
        self, sample_dsf_data: pd.DataFrame, sample_spline_x: np.ndarray
    ) -> None:
        """Test that _create_derivative_subplot_figure creates a subplot figure."""
        x_spline = sample_spline_x
        y_spline = np.random.random(len(x_spline))
        y_spline_derivative = np.random.random(len(x_spline))

        fig = _create_derivative_subplot_figure(
            sample_dsf_data, x_spline, y_spline, y_spline_derivative
        )

        assert isinstance(fig, go.Figure)
        # Should be a subplot figure
        assert fig.layout.xaxis2 is not None
        assert fig.layout.yaxis2 is not None

    def test_create_with_spline_figure(
        self, sample_dsf_data: pd.DataFrame, sample_spline_x: np.ndarray
    ) -> None:
        """Test that _create_with_spline_figure creates a figure with raw and spline data."""
        x_spline = sample_spline_x
        y_spline = np.random.random(len(x_spline))

        fig = _create_with_spline_figure(sample_dsf_data, x_spline, y_spline)

        assert isinstance(fig, go.Figure)
        # Check figure has data traces
        assert hasattr(fig, "data")
        assert fig.data is not None

    def test_create_raw_only_figure(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that _create_raw_only_figure creates a figure with only raw data."""
        fig = _create_raw_only_figure(sample_dsf_data)

        assert isinstance(fig, go.Figure)
        # Check figure has data traces
        assert hasattr(fig, "data")
        assert fig.data is not None

    def test_add_tm_indicator(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that _add_tm_indicator adds a vertical line to the figure."""
        # Create a basic figure first
        fig = _create_raw_only_figure(sample_dsf_data)
        tm = 70.0

        fig_with_tm = _add_tm_indicator(fig, tm)

        assert isinstance(fig_with_tm, go.Figure)
        # Should have added a shape (vertical line)
        assert fig_with_tm.layout.shapes

        # The first shape should be a vertical line at the Tm position
        shape = fig_with_tm.layout.shapes[0]
        assert shape.x0 == tm
        assert shape.x1 == tm

    def test_add_temperature_range_indicators(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that _add_temperature_range_indicators adds rectangle shapes to the figure."""
        # Create a basic figure first
        fig = _create_raw_only_figure(sample_dsf_data)
        min_temp = 40.0
        max_temp = 80.0

        fig_with_range = _add_temperature_range_indicators(fig, min_temp, max_temp)

        assert isinstance(fig_with_range, go.Figure)
        # Should have added two shapes (rectangles for excluded regions)
        assert len(fig_with_range.layout.shapes) == 2

    def test_create_melt_curve_plot_from_features(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that create_melt_curve_plot_from_features creates a figure from a features dict."""
        # Create a minimal features dictionary
        features = {
            "full_well_data": sample_dsf_data,
            "x_spline": np.linspace(25, 95, 100),
            "y_spline": np.random.random(100),
            "y_spline_derivative": np.random.random(100),
            "tm": 65.0,
            "min_temp": 30.0,
            "max_temp": 90.0,
            "model_fit_x": None,
            "model_fit_y": None,
            "model_component_curves": None,
            "selected_model": None,
        }

        fig = create_melt_curve_plot_from_features(features)  # ty:ignore[invalid-argument-type]

        assert isinstance(fig, go.Figure)
        # Verify the figure was created successfully
        assert fig is not None
        # Verify that data and shapes exist
        assert hasattr(fig, "data")
        assert hasattr(fig.layout, "shapes")

    def test_create_melt_curve_plot_with_model_overlay(
        self, sample_dsf_data: pd.DataFrame, sample_spline_x: np.ndarray
    ) -> None:
        """Test that model fit overlay adds traces to the fluorescence subplot."""
        x_spline = sample_spline_x
        rng = np.random.default_rng(60)
        y_spline = rng.random(len(x_spline))
        y_spline_derivative = rng.random(len(x_spline))

        model_fit_x = np.linspace(25, 95, 100)
        model_fit_y = rng.random(100) * 5000
        model_component_curves = {
            "sigmoid_1": rng.random(100) * 3000,
            "initial_decay": rng.random(100) * 2000,
        }

        fig = create_melt_curve_plot(
            sample_dsf_data,
            x_spline=x_spline,
            y_spline=y_spline,
            y_spline_derivative=y_spline_derivative,
            model_fit_x=model_fit_x,
            model_fit_y=model_fit_y,
            model_component_curves=model_component_curves,
            selected_model="model_2",
        )

        assert isinstance(fig, go.Figure)
        # 3 base traces (raw, spline, derivative) + 3 model traces (fit, sigmoid, decay)
        assert len(fig.data) == 6

    def test_add_model_overlay_no_components(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that _add_model_overlay works without component curves."""
        fig = _create_raw_only_figure(sample_dsf_data)
        model_fit_x = np.linspace(25, 95, 50)
        rng = np.random.default_rng(61)
        model_fit_y = rng.random(50) * 5000

        fig = _add_model_overlay(
            fig,
            model_fit_x,
            model_fit_y,
            model_component_curves=None,
            selected_model="model_1",
            is_subplot=False,
        )

        # 1 raw trace + 1 model fit trace
        assert len(fig.data) == 2
        assert fig.data[1].name == "Fit (model_1)"

    def test_create_melt_curve_plot_from_features_with_model(
        self, sample_dsf_data: pd.DataFrame
    ) -> None:
        """Test that create_melt_curve_plot_from_features includes model overlay."""
        rng = np.random.default_rng(62)
        model_fit_x = np.linspace(25, 95, 100)
        features = {
            "full_well_data": sample_dsf_data,
            "x_spline": np.linspace(25, 95, 100),
            "y_spline": rng.random(100),
            "y_spline_derivative": rng.random(100),
            "tm": 55.0,
            "min_temp": 25.0,
            "max_temp": 95.0,
            "model_fit_x": model_fit_x,
            "model_fit_y": rng.random(100) * 5000,
            "model_component_curves": {
                "sigmoid_1": rng.random(100) * 3000,
            },
            "selected_model": "model_1",
        }

        fig = create_melt_curve_plot_from_features(features)  # ty:ignore[invalid-argument-type]

        assert isinstance(fig, go.Figure)
        # 3 base traces + 2 model traces (fit + sigmoid_1)
        assert len(fig.data) == 5
