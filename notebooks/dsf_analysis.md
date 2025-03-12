---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# DSF data analysis with `bada` - a walkthrough

Welcome to a brief demonstration of `bada`, a Python package that allows Biophyiscal Assay Data Analysis! In this notebook, we analyze a 384 well plate generated on QuantStudio 7 (currently supported file formats are Lghtcycler 480 and QuantStudio 7).

Let's start with all the relevant imports:

```python
import pandas as pd
import numpy as np
from pathlib import Path
import bada
from bada.parsers import LightCycler480Parser, QuantStudio7Parser
from bada.processing import get_dsf_curve_features, get_dsf_curve_features_multiple_wells, get_dtw_distances_from_reference
from bada.visualization import create_melt_curve_plot, create_melt_curve_plot_from_features, create_heatmap_plot
from bada.utils.reformatting import convert_distances_to_plate_format, convert_features_to_plate_format
from bada.utils.utils import get_well_ids
```

If you made it here, it means that all required packages have been installed correctly - congratulations! 

## Data parsing
Now, let's get to work and parse the required our data. `bada` offers an easy way to parse data from the machine output files and transform them into a standard format used subsequently in the analysis.

```python
data_path = Path("data")
file_name = "quantstudio7_example.csv"
dsf_data = QuantStudio7Parser(data_path / file_name).parse()
plate_size = 384
```

Now, let's have a quick look at the data that has been parsed. We transformed the data into a dataframe of 3 columns, `well_position`, `temperature` and `fluorescence`.

```python
dsf_data.head()
```

## Control analysis

As a first step, we quickly take a look at our controls. For this particular dataset they are located in the first column whereby the last well is empty. We will also choose a reference control which we will use later several times.

```python
control_wells = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1", "J1", "K1", "L1", "M1", "N1", "O1"]
reference_control = control_wells[0]
```

We can easily view the associated data; let's first only look at the raw data and plot teh measured fluorescence signal against the applied temperature using our reference control

```python
reference_control_data = dsf_data[dsf_data["well_position"] == reference_control]
fig_control = create_melt_curve_plot(reference_control_data)
fig_control.show()
```

The signal looks nice and as expected. For the further analyses, we won't need the entire temperature profile, but focus on the relevant parts (where we expect the maximum first derivative). The data look very smooth, so we won't need a large smoothing factor to create a spline through the data. 

```python
min_temp = 45
max_temp = 65
default_smoothing = 0.01
```

Now we can go ahead and calculate all relevant curve features for our reference control using the `get_dsf_curve_features` function `bada` provides

```python
reference_control_results = get_dsf_curve_features(reference_control_data, min_temp=min_temp, max_temp=max_temp, smoothing=default_smoothing)
```

The results we obtain, can be passed directly to a plotting function `create_melt_curve_plot_from_features` to generate a subplot showing the original raw data, the associated spline as well as the corresponding first derivative. The range at which the spline is calculated is marked by red vertical lines, the position of the maximum first derivative (=`Tm`) is indicated by a vertical orange line:

```python
fig_control_all = create_melt_curve_plot_from_features(reference_control_results)
fig_control_all.show()
```

Now we can go ahead and do the same calculations for all the controls

```python
all_control_results = get_dsf_curve_features_multiple_wells(dsf_data, selected_wells=control_wells, min_temp=min_temp, max_temp=max_temp, smoothing=default_smoothing)
```

We can quickly compare the results we obtain for the controls; in this case we check the `Tm` and the minimum and maximum values of the fluorescence signal.

```python
all_control_results_df = pd.DataFrame.from_dict(all_control_results).loc[['tm', 'min_fluorescence', 'max_fluorescence']].T
all_control_results_df
```

As we observe some variation in the values, let's look at the extreme cases to see if something goes wrong with our analysis. Let's first identify the wells with controls that have the smallest and largest `Tm`, respectively.

```python
control_max_tm = all_control_results_df['tm'].idxmax()
control_min_tm = all_control_results_df['tm'].idxmin()
```

Now we can plot the data for teh control with the largest `Tm`

```python
fig_control_max_tm = create_melt_curve_plot_from_features(all_control_results[control_max_tm])
fig_control_max_tm.show()
```

and also for the minimal `Tm`

```python
fig_control_min_tm = create_melt_curve_plot_from_features(all_control_results[control_min_tm])
fig_control_min_tm.show()
```

So, in the second plot, we see that the derivative shows several peaks; a smoother signal might give us a different result. Generally, it can then be beneficial to increase the smoothing factor slightly, but for this particular dataset that doesn't impact the result (not shown).

In the following, we take this control along anyway and calculate an average `Tm` of all controls which will be used for the `ΔTM` calculations.

```python
control_average_tm = np.mean([v['tm'] for _, v in all_control_results.items()])
control_average_tm
```

## Empty well analysis

If you work with a adatset where you don't have information on where the empty wells where placed, it can be time-consuming to go through all signal profiles and identify them manually. `bada` helps with this by using `Dynamic Time Warping` to compare shapes of all fluorescence signals against teh reference control. If the shapes are similar (but maybe shifted in temperature), they will have a very small difference, while if the shapes are very different, the distances between the cuves are big. To simlify and speed up the analysis, we again work with teh truncated signals: 

```python
m1 = dsf_data['temperature'] >= min_temp
m2 = dsf_data['temperature'] <= max_temp
dsf_data_truncated = dsf_data.loc[m1 & m2, :]
```

`bada` now allows to calculate the distances between the signals and the reference conytrol using `get_dtw_distances_from_reference`. This analysis can run a few seconds

```python
dtw_distances = get_dtw_distances_from_reference(dsf_data_truncated, reference_well=reference_control)
```

The easisest way to inspect the results is a visualization of the results on a heatmap which is straightforward to generate using `create_heatmap_plot`. To convert our data into teh required format, we can use `convert_distances_to_plate_format`

```python
plate_data, cols, rows = convert_distances_to_plate_format(dtw_distances, plate_size)
fig_dtw = create_heatmap_plot(plate_data, cols, rows, title=f"DTW distances to control {reference_control}", colorbar_title="distance measure")
fig_dtw.show()
```

The results seem rather clear: all the blue wells are filled while the red ones are empty. Obviously, if you have prior info on which wells are filled and which ones aren't this analysis is not needed. It is also advisable to always double-check these results. 

## Bulk analysis of all relevant wells
For now we go ahead and filter all wells where we expect to have a meaningful signal (blue wells):

```python
distance_cutoff = 1.0
wells_to_be_analyzed = [k for k, v in dtw_distances.items() if v[0] < distance_cutoff]
```

Now we can do a bulk analysis of all of these wells using `get_dsf_curve_features_multiple_wells`. We also pass the average `Tm` of our controls to calculate all the `ΔTM` values.

```python
all_feature_data = get_dsf_curve_features_multiple_wells(dsf_data, selected_wells=wells_to_be_analyzed, min_temp=min_temp, max_temp=max_temp, avg_control_tm=control_average_tm, smoothing=default_smoothing)
```

Next, let's visualize these `ΔTM` values on a heatmap; we can use the same function as before but again need to reformat the data we just obtained which can easily be achieved by

```python
plate_data_dtm, cols, rows = convert_features_to_plate_format(all_feature_data, plate_size, "delta_tm")
fig_dtm = create_heatmap_plot(plate_data_dtm, cols, rows, title=f"delta Tm using average control Tm as reference", colorbar_title="delta Tm")
fig_dtm.show()
```

So, seems we have a winnder! Well `A15` shows a high `ΔTM`, so let's take a closer look whether the analysis looks solid. We can plot the results easily using the same function as before:

```python
fig_interesting = create_melt_curve_plot_from_features(all_feature_data["A15"])
fig_interesting.show()
```

## Data export

Assuming we checked the results and feel confident about the results, we can easily export all the data. For completeness, we will also include the empty wells. We can identify them using the `DTW distance measures we calculated above:

```python
empty_wells = [k for k, v in dtw_distances.items() if v[0] >= distance_cutoff]
empty_well_info = {
    'tm': np.nan,
    'delta_tm': np.nan,
    'min_fluorescence': np.nan,
    'max_fluorescence': np.nan,
    'fluorescence_range': np.nan,
    'max_derivative_value': np.nan,
    "temp_at_min": np.nan,
    "temp_at_max": np.nan,
    'smoothing': np.nan,
    'min_temp': min_temp,
    'max_temp': min_temp,    
}
all_empty_wells = {well: empty_well_info for well in empty_wells}
```

Let's add the info to our `all_feature_data` dictionary and double-check that we didn't forget any wells:

```python
all_feature_data.update(all_empty_wells)
if len(all_feature_data) != plate_size:
    raise ValueError("something is fishy!")
```

Now we can easily prepare the dataframe for export

```python
export_df = pd.DataFrame.from_dict(all_feature_data).loc[empty_well_info.keys()].T
export_df.head()
```

and then finally export all the results

```python
export_df.to_csv(data_path  / 'dsf_analysis_results.csv')
```
