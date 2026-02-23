import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DSF data analysis with `bada` - a walkthrough

    Welcome to a brief demonstration of `bada`, a Python package that allows Biophysical Assay
    Data Analysis! In this notebook, we analyze a 384 well plate generated on QuantStudio 7
    (currently supported file formats are Lightcycler 480 and QuantStudio 7).

    Let's start with all the relevant imports:
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from bada.parsers import QuantStudio7Parser
    from bada.processing import (
        TmMethod,
        get_dsf_curve_features,
        get_dsf_curve_features_multiple_wells,
        get_dtw_distances_from_reference,
    )
    from bada.utils.reformatting import (
        convert_distances_to_plate_format,
        convert_features_to_plate_format,
    )
    from bada.visualization import (
        create_heatmap_plot,
        create_melt_curve_plot,
        create_melt_curve_plot_from_features,
    )

    return (
        Path,
        QuantStudio7Parser,
        TmMethod,
        convert_distances_to_plate_format,
        convert_features_to_plate_format,
        create_heatmap_plot,
        create_melt_curve_plot,
        create_melt_curve_plot_from_features,
        get_dsf_curve_features,
        get_dsf_curve_features_multiple_wells,
        get_dtw_distances_from_reference,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Data parsing

    Now, let's get to work and parse our data. `bada` offers an easy way to parse data from the
    machine output files and transform them into a standard format used subsequently in the
    analysis.
    """)
    return


@app.cell
def _(Path, QuantStudio7Parser):
    _notebook_dir = Path(__file__).parent
    data_path = _notebook_dir / "data"
    file_name = "quantstudio7_example.csv"
    dsf_data = QuantStudio7Parser(data_path / file_name).parse()
    plate_size = 384
    return data_path, dsf_data, plate_size


@app.cell
def _(mo):
    mo.md(r"""
    Now, let's have a quick look at the data that has been parsed. We transformed the data into
    a dataframe of 3 columns: `well_position`, `temperature` and `fluorescence`.
    """)
    return


@app.cell
def _(dsf_data):
    dsf_data.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Control analysis

    As a first step, we quickly take a look at our controls. For this
    particular dataset they are located in the first column whereby the
    last well is empty. We will also choose a reference control which we
    will use later several times.
    """)
    return


@app.cell
def _():
    control_wells = [
        "A1",
        "B1",
        "C1",
        "D1",
        "E1",
        "F1",
        "G1",
        "H1",
        "I1",
        "J1",
        "K1",
        "L1",
        "M1",
        "N1",
        "O1",
    ]
    reference_control = control_wells[0]
    return control_wells, reference_control


@app.cell
def _(mo):
    mo.md(r"""
    We can easily view the associated data; let's first only look at the raw data and plot the
    measured fluorescence signal against the applied temperature using our reference control:
    """)
    return


@app.cell
def _(create_melt_curve_plot, dsf_data, reference_control):
    reference_control_data = dsf_data[dsf_data["well_position"] == reference_control]
    fig_control = create_melt_curve_plot(reference_control_data)
    fig_control
    return (reference_control_data,)


@app.cell
def _(mo):
    mo.md(r"""
    The signal looks nice and as expected. For the further analyses, we won't need the entire
    temperature profile, but focus on the relevant parts (where we expect the maximum first
    derivative). The data look very smooth, so we won't need a large smoothing factor to create
    a spline through the data.
    """)
    return


@app.cell
def _():
    min_temp = 45
    max_temp = 65
    default_smoothing = 0.01
    return default_smoothing, max_temp, min_temp


@app.cell
def _(mo):
    mo.md(r"""
    ## Single well analysis (derivative method)

    Now we can go ahead and calculate all relevant curve features for our
    reference control using the `get_dsf_curve_features` function. By
    default, Tm is determined from the maximum of the first derivative
    of the B-spline fit:
    """)
    return


@app.cell
def _(
    create_melt_curve_plot_from_features,
    default_smoothing,
    get_dsf_curve_features,
    max_temp,
    min_temp,
    reference_control_data,
):
    reference_control_results = get_dsf_curve_features(
        reference_control_data,
        min_temp=min_temp,
        max_temp=max_temp,
        smoothing=default_smoothing,
    )
    fig_control_all = create_melt_curve_plot_from_features(reference_control_results)
    fig_control_all
    return (reference_control_results,)


@app.cell
def _(mo):
    mo.md(r"""
    The range at which the spline is calculated is marked by red vertical lines, the position of
    the maximum first derivative (= `Tm`) is indicated by a vertical orange line.

    ## Bulk control analysis

    Now we can go ahead and do the same calculations for all the controls:
    """)
    return


@app.cell
def _(
    control_wells,
    default_smoothing,
    dsf_data,
    get_dsf_curve_features_multiple_wells,
    max_temp,
    min_temp,
):
    all_control_result = get_dsf_curve_features_multiple_wells(
        dsf_data,
        selected_wells=control_wells,
        min_temp=min_temp,
        max_temp=max_temp,
        smoothing=default_smoothing,
    )
    return (all_control_result,)


@app.cell
def _(mo):
    mo.md(r"""
    We can quickly compare the results we obtain for the controls; in this case we check the
    `Tm`, the peak confidence, and the minimum and maximum values of the fluorescence signal:
    """)
    return


@app.cell
def _(all_control_result, pd):
    all_control_results_df = pd.DataFrame(
        {
            well: {
                "tm": features["tm"],
                "peak_confidence": features["peak_confidence"],
                "min_fluorescence": features["min_fluorescence"],
                "max_fluorescence": features["max_fluorescence"],
            }
            for well, features in all_control_result.features.items()
        }
    ).T
    all_control_results_df
    return (all_control_results_df,)


@app.cell
def _(mo):
    mo.md(r"""
    As we observe some variation in the values, let's look at the extreme cases to see if
    something goes wrong with our analysis. Let's first identify the wells with controls that
    have the smallest and largest `Tm`, respectively:
    """)
    return


@app.cell
def _(all_control_results_df):
    control_max_tm = all_control_results_df["tm"].idxmax()
    control_min_tm = all_control_results_df["tm"].idxmin()
    return control_max_tm, control_min_tm


@app.cell
def _(mo):
    mo.md(r"""
    Now we can plot the data for the control with the largest `Tm`:
    """)
    return


@app.cell
def _(
    all_control_result,
    control_max_tm,
    create_melt_curve_plot_from_features,
):
    fig_control_max_tm = create_melt_curve_plot_from_features(
        all_control_result.features[control_max_tm]
    )
    fig_control_max_tm
    return


@app.cell
def _(mo):
    mo.md(r"""
    and also for the minimal `Tm`:
    """)
    return


@app.cell
def _(
    all_control_result,
    control_min_tm,
    create_melt_curve_plot_from_features,
):
    fig_control_min_tm = create_melt_curve_plot_from_features(
        all_control_result.features[control_min_tm]
    )
    fig_control_min_tm
    return


@app.cell
def _(mo):
    mo.md(r"""
    So, in the second plot, we see that the derivative shows several peaks; a smoother signal
    might give us a different result. Generally, it can then be beneficial to increase the
    smoothing factor slightly, but for this particular dataset that doesn't impact the result.

    In the following, we take this control along anyway and calculate an average `Tm` of all
    controls which will be used for the `delta Tm` calculations:
    """)
    return


@app.cell
def _(all_control_result, np):
    control_average_tm = float(
        np.mean([features["tm"] for features in all_control_result.features.values()])
    )
    control_average_tm
    return (control_average_tm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Model-based Tm determination

    In addition to the classical derivative-based approach, `bada` implements the
    [DSFworld](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00132) multi-model fitting strategy.
    This approach fits four parametric models (combinations of modified Boltzmann sigmoids with
    optional initial fluorescence decay) and selects the best model via the Bayesian Information
    Criterion (BIC).

    The key advantage: **the model decomposes the melting curve into
    its physical components** (protein unfolding sigmoid(s) and
    dye-related decay), which can yield more accurate Tm values
    especially for non-canonical curve shapes.

    Let's apply this to our reference control using `tm_method="auto"`:
    """)
    return


@app.cell
def _(
    TmMethod,
    create_melt_curve_plot_from_features,
    default_smoothing,
    get_dsf_curve_features,
    max_temp,
    min_temp,
    reference_control_data,
):
    reference_model_results = get_dsf_curve_features(
        reference_control_data,
        min_temp=min_temp,
        max_temp=max_temp,
        smoothing=default_smoothing,
        tm_method=TmMethod.AUTO,
    )
    fig_model = create_melt_curve_plot_from_features(reference_model_results)
    fig_model
    return (reference_model_results,)


@app.cell
def _(mo):
    mo.md(r"""
    The plot now shows the model fit overlay (red solid line) and its component curves (dashed
    lines). The orange vertical line indicates the Tm derived from the model.

    Let's compare the derivative-based and model-based results:
    """)
    return


@app.cell
def _(mo, reference_control_results, reference_model_results):
    _d = reference_control_results
    _m = reference_model_results
    mo.md(
        f"""
    | Metric | Derivative | Model |
    |--------|----------:|------:|
    | **Tm** | {_d["tm"]:.2f} °C | {_m["tm"]:.2f} °C |
    | **Selected model** | — | {_m["selected_model"]} |
    | **R²** | — | {_m["model_r_squared"]:.4f} |
    | **Peak confidence** | {_d["peak_confidence"]:.2f} | {_m["peak_confidence"]:.2f} |
        """
    )
    return


@app.cell
def _(mo, pd, reference_model_results):
    bic_df = pd.DataFrame(
        [
            {"Model": model, "BIC": round(bic, 1)}
            for model, bic in reference_model_results["bic_values"].items()
        ]
    ).set_index("Model")
    mo.vstack([mo.md("**BIC values per model** (lower is better):"), bic_df])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Bulk model fitting on controls

    Let's compare derivative vs. model-based Tm for all control wells:
    """)
    return


@app.cell
def _(
    TmMethod,
    control_wells,
    default_smoothing,
    dsf_data,
    get_dsf_curve_features_multiple_wells,
    max_temp,
    min_temp,
):
    all_control_model_result = get_dsf_curve_features_multiple_wells(
        dsf_data,
        selected_wells=control_wells,
        min_temp=min_temp,
        max_temp=max_temp,
        smoothing=default_smoothing,
        tm_method=TmMethod.AUTO,
    )
    return (all_control_model_result,)


@app.cell
def _(all_control_model_result, all_control_result, pd):
    comparison_df = pd.DataFrame(
        {
            well: {
                "tm_derivative": all_control_result.features[well]["tm"],
                "tm_model": all_control_model_result.features[well]["tm"],
                "delta": (
                    all_control_model_result.features[well]["tm"]
                    - all_control_result.features[well]["tm"]
                ),
                "selected_model": all_control_model_result.features[well]["selected_model"],
                "r_squared": all_control_model_result.features[well]["model_r_squared"],
            }
            for well in all_control_result.features
            if well in all_control_model_result.features
        }
    ).T
    comparison_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Identify atypical behavior

    If you work with a dataset where you don't have information on
    where the empty wells were placed, it can be time-consuming to go
    through all signal profiles and identify them manually. And even if
    you know which ones are filled, there might be cases where something
    went wrong leading to a signal with an atypical shape.

    `bada` helps with this by using **Dynamic Time Warping (DTW)** to compare shapes of all
    fluorescence signals against the reference control. If the shapes are similar (but maybe
    shifted in temperature), they will have a very small distance, while very different shapes
    produce large distances.

    To simplify and speed up the analysis, we work with the truncated signals:
    """)
    return


@app.cell
def _(dsf_data, max_temp, min_temp):
    m1 = dsf_data["temperature"] >= min_temp
    m2 = dsf_data["temperature"] <= max_temp
    dsf_data_truncated = dsf_data.loc[m1 & m2, :]
    return (dsf_data_truncated,)


@app.cell
def _(dsf_data_truncated, get_dtw_distances_from_reference, reference_control):
    dtw_distances = get_dtw_distances_from_reference(
        dsf_data_truncated, reference_well=reference_control
    )
    return (dtw_distances,)


@app.cell
def _(mo):
    mo.md(r"""
    The easiest way to inspect the results is a visualization on a
    heatmap. To convert our data into the required format, we can use
    `convert_distances_to_plate_format`:
    """)
    return


@app.cell
def _(
    convert_distances_to_plate_format,
    create_heatmap_plot,
    dtw_distances,
    plate_size,
    reference_control,
):
    plate_data_dtw, cols_dtw, rows_dtw = convert_distances_to_plate_format(
        dtw_distances, plate_size
    )
    fig_dtw = create_heatmap_plot(
        plate_data_dtw,
        cols_dtw,
        rows_dtw,
        title=f"DTW distances to control {reference_control}",
        colorbar_title="distance measure",
    )
    fig_dtw
    return


@app.cell
def _(mo):
    mo.md(r"""
    The results seem rather clear: all the blue wells are filled while the red ones are empty.
    Obviously, if you have prior info on which wells are filled and which ones aren't this
    analysis is not needed, however, it can still be useful to identify atypical shapes and to
    exclude the corresponding wells from further analysis.

    ## Bulk analysis of all relevant wells

    For now we go ahead and filter all wells where we expect to have a meaningful signal
    (blue wells). You can adjust the DTW distance cutoff:
    """)
    return


@app.cell
def _(mo):
    distance_cutoff_slider = mo.ui.slider(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=1.0,
        label="DTW distance cutoff",
    )
    distance_cutoff_slider
    return (distance_cutoff_slider,)


@app.cell
def _(distance_cutoff_slider, dtw_distances, mo):
    distance_cutoff = distance_cutoff_slider.value
    wells_to_be_analyzed = [k for k, v in dtw_distances.items() if v[0] < distance_cutoff]
    mo.md(
        f"**{len(wells_to_be_analyzed)}** wells selected for analysis (cutoff = {distance_cutoff})"
    )
    return distance_cutoff, wells_to_be_analyzed


@app.cell
def _(mo):
    mo.md(r"""
    Now we can do a bulk analysis of all of these wells using
    `get_dsf_curve_features_multiple_wells`. We also pass the average `Tm` of our controls to
    calculate all the `delta Tm` values:
    """)
    return


@app.cell
def _(
    control_average_tm,
    default_smoothing,
    dsf_data,
    get_dsf_curve_features_multiple_wells,
    max_temp,
    min_temp,
    wells_to_be_analyzed,
):
    all_feature_result = get_dsf_curve_features_multiple_wells(
        dsf_data,
        selected_wells=wells_to_be_analyzed,
        min_temp=min_temp,
        max_temp=max_temp,
        avg_control_tm=control_average_tm,
        smoothing=default_smoothing,
    )
    return (all_feature_result,)


@app.cell
def _(mo):
    mo.md(r"""
    Next, let's visualize these `delta Tm` values on a heatmap:
    """)
    return


@app.cell
def _(
    all_feature_result,
    convert_features_to_plate_format,
    create_heatmap_plot,
    plate_size,
):
    plate_data_dtm, cols_dtm, rows_dtm = convert_features_to_plate_format(
        all_feature_result.features, plate_size, "delta_tm"
    )
    fig_dtm = create_heatmap_plot(
        plate_data_dtm,
        cols_dtm,
        rows_dtm,
        title="delta Tm using average control Tm as reference",
        colorbar_title="delta Tm",
    )
    fig_dtm
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hit validation

    So, seems we have a winner! Well `A15` shows a high `delta Tm`, so let's take a closer
    look whether the analysis looks solid.

    We can inspect the derivative-based analysis:
    """)
    return


@app.cell
def _(all_feature_result, create_melt_curve_plot_from_features):
    fig_interesting = create_melt_curve_plot_from_features(all_feature_result.features["A15"])
    fig_interesting
    return


@app.cell
def _(mo):
    mo.md(r"""
    And now let's also look at the model-based analysis for the same well to compare approaches:
    """)
    return


@app.cell
def _(
    TmMethod,
    control_average_tm,
    create_melt_curve_plot_from_features,
    default_smoothing,
    dsf_data,
    get_dsf_curve_features,
    max_temp,
    min_temp,
):
    a15_data = dsf_data[dsf_data["well_position"] == "A15"]
    a15_model_results = get_dsf_curve_features(
        a15_data,
        min_temp=min_temp,
        max_temp=max_temp,
        smoothing=default_smoothing,
        avg_control_tm=control_average_tm,
        tm_method=TmMethod.AUTO,
    )
    fig_a15_model = create_melt_curve_plot_from_features(a15_model_results)
    fig_a15_model
    return (a15_model_results,)


@app.cell
def _(a15_model_results, all_feature_result, mo):
    a15_deriv = all_feature_result.features["A15"]
    mo.md(
        f"""
        **A15 comparison:**

        | Metric | Derivative | Model |
        |--------|----------:|------:|
        | **Tm** | {a15_deriv["tm"]:.2f} °C | {a15_model_results["tm"]:.2f} °C |
        | **delta Tm** | {a15_deriv["delta_tm"]:.2f} °C | {a15_model_results["delta_tm"]:.2f} °C |
        | **Selected model** | — | {a15_model_results["selected_model"]} |
        | **R²** | — | {a15_model_results["model_r_squared"]:.4f} |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Peak detection quality

    The robust peak detection pipeline provides confidence scores and quality flags for each
    well. Let's inspect these for a few representative wells:
    """)
    return


@app.cell
def _(all_feature_result, mo, pd):
    quality_wells = ["A1", "A15", "C1", "M1"]
    quality_df = pd.DataFrame(
        {
            well: {
                "tm": all_feature_result.features[well]["tm"],
                "peak_confidence": all_feature_result.features[well]["peak_confidence"],
                "peak_is_valid": all_feature_result.features[well]["peak_is_valid"],
                "n_peaks_detected": all_feature_result.features[well]["n_peaks_detected"],
                "quality_flags": ", ".join(all_feature_result.features[well]["peak_quality_flags"])
                or "none",
            }
            for well in quality_wells
            if well in all_feature_result.features
        }
    ).T
    mo.md("### Peak detection quality for selected wells")
    return (quality_df,)


@app.cell
def _(quality_df):
    quality_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data export

    Assuming we checked the results and feel confident about them, we can easily export all the
    data. For completeness, we will also include the empty wells:
    """)
    return


@app.cell
def _(
    all_feature_result,
    distance_cutoff,
    dtw_distances,
    min_temp,
    np,
    pd,
    plate_size,
):
    empty_wells = [k for k, v in dtw_distances.items() if v[0] >= distance_cutoff]
    empty_well_info = {
        "tm": np.nan,
        "delta_tm": np.nan,
        "min_fluorescence": np.nan,
        "max_fluorescence": np.nan,
        "fluorescence_range": np.nan,
        "max_derivative_value": np.nan,
        "temp_at_min": np.nan,
        "temp_at_max": np.nan,
        "smoothing": np.nan,
        "min_temp": min_temp,
        "max_temp": min_temp,
    }

    # Combine filled and empty wells
    export_features = dict(all_feature_result.features)
    export_features.update({well: empty_well_info for well in empty_wells})

    if len(export_features) != plate_size:
        raise ValueError(f"Expected {plate_size} wells, got {len(export_features)}")

    export_df = pd.DataFrame(
        {
            well: {k: features[k] for k in empty_well_info}
            for well, features in export_features.items()
        }
    ).T
    export_df
    return (export_df,)


@app.cell
def _(data_path, export_df):
    export_df.to_csv(data_path / "dsf_analysis_results.csv")
    return


if __name__ == "__main__":
    app.run()
