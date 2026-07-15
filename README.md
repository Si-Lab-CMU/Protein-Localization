# E. coli Fluorescence Line-Profile Analysis and Membrane-Fraction Fitting

This repository contains a Python workflow for analyzing agar-pad fluorescence microscopy images of *E. coli* cells and estimating the fraction of fluorescent protein localized to the membrane. The workflow starts from raw Nikon `.nd2` image files, extracts single-cell fluorescence line profiles along the short axis of each cell, computes an average profile for each sample, and then fits unknown samples as a linear mixture of a 100% membrane reference profile and a 100% cytoplasmic reference profile.

The intended use case is to compare calibration/reference strains and strains with unknown protein distributions. For example, a calibration strain or reference sample can be used to define the pure membrane profile, another reference sample can define the pure cytoplasmic profile, and unknown samples are then fitted to estimate the membrane fraction \(\alpha\).

## Workflow overview

```mermaid
flowchart LR
    A[Raw fluorescence/phase images<br/>1_data/*.nd2] --> B[runagar.py / agarPad.py<br/>fluorescence analysis]
    B --> C[3_analysis<br/>intensity .txt files<br/>line_profile .pkl files]
    C --> D[lineprofile analysis.ipynb<br/>align and average all-cell line profiles]
    D --> E[7_line profile figs with avg line_x=[-18,18]<br/>average line-profile plots]
    D --> F[8_lineprofile_data<br/>x_ref.npy and y_mean.npy]
    F --> G[membrane fraction linear fitting_least square.ipynb<br/>linear mixture fitting]
    G --> H[10_fitting_plots<br/>fitting plots and estimated alpha]
```

## Repository structure

```text
.
├── 1_data/
│   └── raw .nd2 image files should be placed here
├── 2_code/
│   ├── runagar.py                    # batch converts .nd2 to .tif and runs agarPad.py
│   ├── agarPad.py                    # main cell segmentation / fluorescence analysis script
│   ├── APLineProfile.py              # short-axis fluorescence line-profile extraction
│   ├── APFluorescence.py             # cell fluorescence intensity calculation
│   ├── APPhase.py, APLibrary.py      # phase-contrast segmentation and contour utilities
│   ├── util_local.py                 # local file/folder helper functions
│   └── env/
│       ├── agarpad.yml
│       └── agarpad-mo.yml            # conda environment files
├── 3_analysis/
│   ├── *_int.txt                     # fluorescence intensity results
│   ├── *_pc_thr.txt                  # phase/contour size results
│   └── *_line_profile.pkl            # all-cell line-profile data
├── 7_line profile figs with avg line_x=[-18,18]/
│   └── *_fit.png                     # average line-profile plots
├── 8_lineprofile_data/
│   ├── *_x_ref.npy                   # common x axis for averaged profile
│   └── *_y_mean.npy                  # mean line-profile intensity
├── 10_fitting_plots/
│   └── *_fit.png                     # membrane-fraction fitting plots
└── analysis_notebook/
    ├── lineprofile analysis.ipynb
    └── membrane fraction linear fitting_least square.ipynb
```

> **Note:** The current notebooks use relative paths. Run `runagar.py` from inside `2_code/`, and run the notebooks from inside `analysis_notebook/`, unless you update the paths manually.

## Installation

The analysis was developed in Python and uses `numpy`, `scipy`, `matplotlib`, `scikit-image`, `opencv`, `nd2`, `tifffile`, `pyyaml`, `kmeans1d`, and Jupyter Notebook.

A conda environment file is included. For example:

```bash
conda env create -f 2_code/env/agarpad-mo.yml
conda activate agarpad-mo
```

If the provided environment is too platform-specific, install the required packages manually:

```bash
conda create -n agarpad python=3.9
conda activate agarpad
pip install numpy scipy matplotlib scikit-image opencv-python nd2 tifffile pyyaml kmeans1d termcolor notebook
```

## Step 1: Analyze raw microscopy images

Place raw `.nd2` files in `1_data/`. Then run:

```bash
cd 2_code
python runagar.py
```

`runagar.py` performs two main tasks:

1. Converts each `.nd2` file into `.tif` images. For multi-channel images, each field of view and channel is saved as a separate TIFF file using names like:

```text
<basename>_XY000_C1.tif
<basename>_XY000_C2.tif
```

2. Automatically generates parameter files and runs `agarPad.py` on each image folder.

The current default parameter template in `runagar.py` assumes:

```python
channels = ['pc_thr', 'int']
channel_postfixes = ['C2', 'C1']
pxl2um = 0.065
```

This means that `C2` is treated as the phase-contrast channel for segmentation and `C1` is treated as the fluorescence channel for intensity and line-profile analysis. If your imaging channel order is different, update `channel_postfixes` in `runagar.py` before running the analysis.

### Step 1 outputs

The main output folder is `3_analysis/`. Important files include:

| File type | Description |
|---|---|
| `*_pc_thr.txt` | Cell size and contour information from phase-contrast analysis |
| `*_int.txt` | Fluorescence intensity measurements for each cell |
| `*_line_profile.pkl` | All single-cell short-axis fluorescence line profiles |

The `*_int.txt` file contains cell index, average fluorescence intensity, background-subtracted total fluorescence intensity, image background estimate, and volume-normalized fluorescence intensity. The `*_line_profile.pkl` files are used in the next analysis step.

## Step 2: Generate average line profiles

Open:

```text
analysis_notebook/lineprofile analysis.ipynb
```

This notebook reads all `.pkl` files from `3_analysis/`, aligns individual cell profiles, interpolates them onto a common x-axis, plots all individual profiles, and calculates the average line profile.

The key function is `plot_pickle_data(...)`. For each cell profile, the notebook:

1. Reads the short-axis fluorescence profile from the `.pkl` file.
2. Finds the profile center using a Gaussian fit. If the fit fails, it falls back to the maximum-intensity position.
3. Shifts each profile so the fitted center is at zero.
4. Interpolates each profile onto a common x-axis, currently typically set by:

```python
aligned_x_range = (-15, 15)
num_points = 150
```

5. Averages all valid profiles with `np.nanmean`.
6. Saves the average profile as `.npy` files.

### Step 2 outputs

The expected outputs are:

```text
7_line profile figs with avg line_x=[-18,18]/
8_lineprofile_data/
```

The files in `8_lineprofile_data/` are the inputs for membrane-fraction fitting:

```text
<sample>_x_ref.npy
<sample>_y_mean.npy
```

`x_ref.npy` stores the common x-axis and `y_mean.npy` stores the average line-profile intensity.

> **Path note:** Some notebook cells may still save to older folder names such as `../line profile figs with avg line` or `../3_analysis_narmolized`. If you want the output to match this repository structure, update those paths to `../7_line profile figs with avg line_x=[-18,18]` and `../8_lineprofile_data`.

## Step 3: Estimate membrane fraction by linear mixture fitting

Open:

```text
analysis_notebook/membrane fraction linear fitting_least square.ipynb
```

This notebook estimates the membrane fraction \(\alpha\) of an unknown sample using a 100% membrane reference profile and a 100% cytoplasmic reference profile.

The fitting model is:

$$
L_{\mathrm{unknown}}(x) = \alpha L_{\mathrm{mem}}(x) + (1-\alpha)L_{\mathrm{cyto}}(x),
$$

where:

- \(L_{\mathrm{unknown}}(x)\) is the measured average line profile of the unknown sample.
- \(L_{\mathrm{mem}}(x)\) is the 100% membrane reference profile.
- \(L_{\mathrm{cyto}}(x)\) is the 100% cytoplasmic reference profile.
- \(\alpha\) is the fitted membrane fraction.

The notebook includes a least-squares fitting function that scans \(\alpha\) from 0 to 1 and finds the value minimizing:

$$
\mathrm{MSE}(\alpha) = \frac{1}{n}\sum_i \left[\alpha L_{\mathrm{mem}}(x_i) + (1-\alpha)L_{\mathrm{cyto}}(x_i) - L_{\mathrm{unknown}}(x_i)\right]^2.
$$

### Example usage

```python
folder = "../8_lineprofile_data"

alpha = fit_alpha_from_profile(
    folder=folder,
    membrane_x_file="MZ029_100uMIPTG_C1_pc_thr_line_profile_x_ref.npy",
    membrane_y_file="MZ029_100uMIPTG_C1_pc_thr_line_profile_y_mean.npy",
    cyto_x_file="MZ001_20ngCTC_100uMIPTG_C2_pc_thr_line_profile_x_ref.npy",
    cyto_y_file="MZ001_20ngCTC_100uMIPTG_C2_pc_thr_line_profile_y_mean.npy",
    unknown_x_file="FS460_C1_pc_thr_line_profile_x_ref.npy",
    unknown_y_file="FS460_C1_pc_thr_line_profile_y_mean.npy"
)

print(f"Estimated alpha: {alpha:.4f}")
```

The fitting plots are saved to:

```text
10_fitting_plots/
```

## Choosing reference profiles

To estimate a meaningful membrane fraction, choose reference profiles carefully:

- `100% membrane reference profile`: a strain or condition where the fluorescent protein is expected to be fully membrane-localized.
- `100% cytoplasm reference profile`: a strain or condition where the fluorescent protein is expected to be cytoplasmic.
- `unknown sample profile`: the strain or condition whose membrane fraction is being estimated.

All three profiles should be processed through the same segmentation, line-profile extraction, alignment, and averaging pipeline.

## Important assumptions

The linear mixture model assumes that fluorescence imaging is approximately linear: the measured profile from a mixed distribution is the sum of the profiles contributed by membrane-localized and cytoplasmic proteins. Under this assumption, the unknown profile can be represented as a weighted sum of the two reference profiles.

For \(\alpha\) to be interpreted as a molecular membrane fraction, the reference profiles should be on a comparable fluorescence scale. If each profile is independently normalized to its own maximum, the fitted \(\alpha\) mainly reflects profile shape similarity rather than an absolute molecular fraction.

## Practical notes

- Make sure the raw `.nd2` files are placed in `1_data/` before running `runagar.py`.
- Confirm the channel order in `runagar.py` before analysis. The default assumes phase contrast is `C2` and fluorescence is `C1`.
- The pixel size is set by `pxl2um = 0.065` in the auto-generated parameter template. Update this if your microscope calibration is different.
- The line-profile analysis notebook expects `.pkl` files from `3_analysis/`.
- The fitting notebook expects paired `_x_ref.npy` and `_y_mean.npy` files from `8_lineprofile_data/`.
- If two profile files have different x-axes, use interpolation-based fitting. If all profiles were already generated on the same x-axis, a no-interpolation version can be used.
- Avoid committing large raw image files to GitHub. Use a small example dataset or provide a download link for raw data.

## Recommended cleanup before uploading to GitHub

Before publishing the repository, consider removing generated or system-specific files:

```text
.DS_Store
__MACOSX/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

A suggested `.gitignore` is:

```gitignore
.DS_Store
__MACOSX/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.nd2
*.tif
```

If you want to include example data, keep only a small test dataset so users can run the pipeline quickly.

## Troubleshooting

### No cells are detected

Check the contour filtering parameters in the parameter template generated by `runagar.py`, especially `min_area`, `boundary`, `con_min_w`, `con_max_w`, `con_min_l`, and `con_max_l`.

### The wrong channel is analyzed

Update the following lines in `runagar.py`:

```python
channels = ['pc_thr', 'int']
channel_postfixes = ['C2', 'C1']
```

The first postfix corresponds to phase/segmentation and the second postfix corresponds to fluorescence intensity/line-profile analysis.

### Line-profile figures are not saved to the expected folder

Check the `plot_dir` and `data_dir` variables inside `analysis_notebook/lineprofile analysis.ipynb`.

### Fitting results look inconsistent

Check that the membrane, cytoplasm, and unknown profiles were generated using the same alignment range, number of interpolation points, and preprocessing settings. Also confirm whether the fitting function normalizes each profile before fitting, because normalization changes the interpretation of \(\alpha\).

## Output summary

| Step | Input | Script / Notebook | Output |
|---|---|---|---|
| 1 | `1_data/*.nd2` | `2_code/runagar.py` | `3_analysis/*_int.txt`, `*_pc_thr.txt`, `*_line_profile.pkl` |
| 2 | `3_analysis/*_line_profile.pkl` | `analysis_notebook/lineprofile analysis.ipynb` | `7_line profile figs.../*.png`, `8_lineprofile_data/*_x_ref.npy`, `*_y_mean.npy` |
| 3 | `8_lineprofile_data/*.npy` | `analysis_notebook/membrane fraction linear fitting_least square.ipynb` | estimated \(\alpha\), fitting plots in `10_fitting_plots/` |
