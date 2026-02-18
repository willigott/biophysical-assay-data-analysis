from enum import Enum

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bada.processing.feature_extraction import DSFCurveFeatures
from bada.utils.validation import validate_temperature_range


class PlotType(Enum):
    """Enum representing the different types of melting curve plots."""

    DERIVATIVE = "derivative"
    SPLINE = "spline"
    RAW = "raw"


def _infer_plot_type(
    x_spline: np.ndarray | None,
    y_spline: np.ndarray | None,
    y_spline_derivative: np.ndarray | None,
) -> PlotType:
    """Determine the appropriate plot type based on available data.

    Args:
        x_spline: Temperature values for spline fit or None
        y_spline: Fluorescence values for spline fit or None
        y_spline_derivative: First derivative values or None

    Returns:
        PlotType: The inferred plot type based on available data
    """
    has_derivative = y_spline_derivative is not None
    has_spline = x_spline is not None and y_spline is not None
    if has_derivative and has_spline:
        return PlotType.DERIVATIVE
    elif has_spline:
        return PlotType.SPLINE
    else:
        return PlotType.RAW


def create_melt_curve_plot(
    full_well_data: pd.DataFrame,
    x_spline: np.ndarray | None = None,
    y_spline: np.ndarray | None = None,
    y_spline_derivative: np.ndarray | None = None,
    tm: float | None = None,
    min_temp: float | None = None,
    max_temp: float | None = None,
) -> go.Figure:
    """Create a melt curve plot with automatic display options based on provided data.

    The plot type is automatically determined based on the data provided:
    - If y_spline_derivative is provided: Show melt curve with derivative subplot
    - If only x_spline and y_spline are provided: Show melt curve with spline fit
    - If only full_well_data is provided: Show just the raw data

    Args:
        full_well_data: DataFrame containing raw melting curve data
        x_spline: Optional temperature values for spline fit
        y_spline: Optional fluorescence values for spline fit
        y_spline_derivative: Optional first derivative values of the spline fit
        tm: Optional melting temperature
        min_temp: Optional minimum temperature to highlight
        max_temp: Optional maximum temperature to highlight

    Returns:
        plotly.graph_objects.Figure
    """
    show_temp_range = validate_temperature_range(min_temp, max_temp)

    plot_type = _infer_plot_type(x_spline, y_spline, y_spline_derivative)

    if plot_type == PlotType.DERIVATIVE:
        # we know these are not None based on the inferred plot type
        fig = _create_derivative_subplot_figure(
            full_well_data,
            x_spline,  # type: ignore
            y_spline,  # type: ignore
            y_spline_derivative,  # type: ignore
        )
    elif plot_type == PlotType.SPLINE:
        fig = _create_with_spline_figure(
            full_well_data,
            x_spline,  # type: ignore
            y_spline,  # type: ignore
        )
    else:  # plot_type == PlotType.RAW
        fig = _create_raw_only_figure(full_well_data)

    # add temperature indicators independently
    if tm is not None:
        fig = _add_tm_indicator(fig, tm)

    if show_temp_range:
        # we know min_temp and max_temp are not None if show_temp_range is True
        fig = _add_temperature_range_indicators(fig, min_temp, max_temp)  # type: ignore

    return fig


def _create_derivative_subplot_figure(
    full_well_data: pd.DataFrame,
    x_spline: np.ndarray,
    y_spline: np.ndarray,
    y_spline_derivative: np.ndarray,
) -> go.Figure:
    """Create a figure with melt curve and first derivative subplots.

    Args:
        full_well_data: DataFrame containing raw data
        x_spline: Temperature values for spline fit
        y_spline: Fluorescence values for spline fit
        y_spline_derivative: First derivative values

    Returns:
        plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Melt curve", "First derivative"),
    )

    # Add raw data to top subplot
    fig.add_trace(
        go.Scatter(
            x=full_well_data["temperature"],
            y=full_well_data["fluorescence"],
            name="Raw Data",
            mode="markers+lines",
        ),
        row=1,
        col=1,
    )

    # Add spline fit to top subplot
    fig.add_trace(
        go.Scatter(
            x=x_spline, y=y_spline, name="Spline Fit", line=dict(color="green", dash="dash")
        ),
        row=1,
        col=1,
    )

    # Add derivative to bottom subplot
    fig.add_trace(
        go.Scatter(
            x=x_spline, y=y_spline_derivative, name="First Derivative", line=dict(color="blue")
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis2_title="Temperature (°C)",
        yaxis_title="Fluorescence",
        yaxis2_title="dF/dT",
    )

    return fig


def _create_with_spline_figure(
    full_well_data: pd.DataFrame, x_spline: np.ndarray, y_spline: np.ndarray
) -> go.Figure:
    """Create a figure with raw data and spline fit.

    Args:
        full_well_data: DataFrame containing raw data
        x_spline: Temperature values for spline fit
        y_spline: Fluorescence values for spline fit

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Add raw data
    fig.add_trace(
        go.Scatter(
            x=full_well_data["temperature"],
            y=full_well_data["fluorescence"],
            name="Raw Data",
            mode="markers+lines",
        )
    )

    # Add spline fit
    fig.add_trace(
        go.Scatter(x=x_spline, y=y_spline, name="Spline Fit", line=dict(color="green", dash="dash"))
    )

    fig.update_layout(
        height=500, showlegend=True, xaxis_title="Temperature (°C)", yaxis_title="Fluorescence"
    )

    return fig


def _create_raw_only_figure(full_well_data: pd.DataFrame) -> go.Figure:
    """Create a figure with only raw data.

    Args:
        full_well_data: DataFrame containing raw data

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Add only raw data
    fig.add_trace(
        go.Scatter(
            x=full_well_data["temperature"],
            y=full_well_data["fluorescence"],
            name="Raw Data",
            mode="markers+lines",
        )
    )

    fig.update_layout(
        height=500, showlegend=True, xaxis_title="Temperature (°C)", yaxis_title="Fluorescence"
    )

    return fig


def _add_tm_indicator(fig: go.Figure, tm: float) -> go.Figure:
    """Add melting temperature indicator line to the figure.

    Args:
        fig: Plotly figure to add line to
        tm: Melting temperature

    Returns:
        plotly.graph_objects.Figure with Tm line added
    """

    fig.add_vline(
        x=tm, line_dash="dash", line_color="orange", annotation_text=f"Tm = {tm:.2f}°C", row="all"
    )

    return fig


def _add_temperature_range_indicators(
    fig: go.Figure, min_temp: float, max_temp: float
) -> go.Figure:
    """Add minimum and maximum temperature indicator lines to the figure.

    Args:
        fig: Plotly figure to add lines to
        min_temp: Minimum temperature
        max_temp: Maximum temperature

    Returns:
        plotly.graph_objects.Figure with temperature range lines added
    """
    # Add min_temp line
    fig.add_vline(
        x=min_temp,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Min Temp = {min_temp:.1f}°C",
        row="all",
    )

    # Add max_temp line
    fig.add_vline(
        x=max_temp,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max Temp = {max_temp:.1f}°C",
        row="all",
    )

    return fig


def create_melt_curve_plot_from_features(
    curve_features: DSFCurveFeatures,
) -> go.Figure:
    """
    Create a melt curve plot from the results of get_dsf_curve_features

    Args:
        curve_features: Dictionary returned by get_dsf_curve_features
        min_temp: Optional override for min temperature
        max_temp: Optional override for max temperature
        **additional_plot_kwargs: Additional keyword arguments to pass to create_melt_curve_plot

    Returns:
        plt.Figure: The generated figure
    """
    plot_data = {
        "full_well_data": curve_features["full_well_data"],
        "x_spline": curve_features["x_spline"],
        "y_spline": curve_features["y_spline"],
        "y_spline_derivative": curve_features["y_spline_derivative"],
        "tm": curve_features["tm"],
        "min_temp": curve_features["min_temp"],
        "max_temp": curve_features["max_temp"],
    }

    return create_melt_curve_plot(**plot_data)
