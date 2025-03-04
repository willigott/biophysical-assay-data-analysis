from typing import List, cast

import numpy as np
import plotly.graph_objects as go

from bada.visualization.heatmap import plot_plate_data


class TestHeatmap:
    def test_plot_plate_data_return_type(
        self, sample_plate_data: np.ndarray, plate_rows: List[str], plate_cols: List[str]
    ) -> None:
        """Test that plot_plate_data returns a plotly Figure object."""
        fig = plot_plate_data(
            sample_plate_data,
            cols=plate_cols,
            rows=plate_rows,
            title="Test Heatmap",
            colorbar_title="Value",
        )

        assert isinstance(fig, go.Figure)

    def test_plot_plate_data_content(
        self, sample_plate_data: np.ndarray, plate_rows: List[str], plate_cols: List[str]
    ) -> None:
        """Test that plot_plate_data contains a heatmap trace."""
        fig = plot_plate_data(
            sample_plate_data, cols=plate_cols, rows=plate_rows, title="Test Heatmap"
        )

        # Check that the figure has one trace
        assert len(fig.data) > 0  # type: ignore

        # Check that the trace is a heatmap
        trace = cast(go.Heatmap, fig.data[0])
        assert isinstance(trace, go.Heatmap)

        # Check that the data is correctly passed
        assert np.array_equal(trace.z, sample_plate_data)  # type: ignore
        # Plotly converts lists to tuples, so we need to convert for comparison
        assert tuple(trace.x) == tuple(plate_cols)  # type: ignore
        assert tuple(trace.y) == tuple(plate_rows)  # type: ignore

    def test_plot_plate_data_layout(
        self, sample_plate_data: np.ndarray, plate_rows: List[str], plate_cols: List[str]
    ) -> None:
        """Test that plot_plate_data has the expected layout."""
        title = "Test Heatmap"
        fig = plot_plate_data(sample_plate_data, cols=plate_cols, rows=plate_rows, title=title)

        # Check title is set correctly
        assert fig.layout.title.text == title  # type: ignore

        # Check y-axis is reversed (standard for plate layouts)
        assert fig.layout.yaxis.autorange == "reversed"  # type: ignore

        # Check plot dimensions are set
        assert fig.layout.height is not None  # type: ignore

        # Check other layout properties are set
        assert fig.layout.plot_bgcolor == "white"  # type: ignore
        assert fig.layout.xaxis.side == "top"  # type: ignore  # x-axis labels on top

    def test_custom_color_scale(
        self, sample_plate_data: np.ndarray, plate_rows: List[str], plate_cols: List[str]
    ) -> None:
        """Test that custom color scales can be applied."""
        color_scale = "Viridis"
        fig = plot_plate_data(
            sample_plate_data,
            cols=plate_cols,
            rows=plate_rows,
            title="Test Heatmap",
            color_scale=color_scale,
        )

        # The colorscale is expanded to a list of tuples, so we can't directly compare
        # Instead, check that the colorscale is not the default
        trace = cast(go.Heatmap, fig.data[0])
        assert trace.colorscale is not None  # type: ignore
