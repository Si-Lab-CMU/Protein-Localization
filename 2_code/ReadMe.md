## Usage

- Update the `path_to_tiffs`, `path_analysis`, and `path_debug` to local path.
- Run the code in terminal as `python PathTo/agarPad.py -f PathTo/agarPad_params.yaml`.

## Analysis file annotation:

'*pc_thr.txt': analysis results for phase contract images. 

    - cell index
    - cell length in um
    - cell width in um
    - cell area in um2
    - cell perimeter in pixels
    - cell bright filed average values averged in pixel numbers

'*int.txt': analysis results for fluorescent intensity images.

    - cell index
    - cell average intensity for each pixel (background not substracted)
    - cell total intensity for all pixels (background subtracted)
    - image mean background for each pixel
    - cell average intensity for volume (background subtracted, obtained from total intensity divided by total volume (um3))

## Note

**params.yaml**: this is the parameter file. Note, for path parameters, such as `path_to_tiffs`, it is necessary to end the folder path with "/", in order to run the program correctly. 

## Updates

#### 20230131: Background mean algorithm update and bug fix

- **APLuorescence.py**: Updated background mean calculation for fluorescent intensity. 1D K-means clustering was applied and then the background mean is calculated as lower centroid. 

- **agarPad.py**: Updated the unordered file list. Allows automatically detect and generate the debug folder (`path_analysis`) and analysis folder (`path_debug`) defined in the parameter file.

#### 20230131:

- **agarPad.py**: Updated calculation for volumn averge intensity, in unit of AU/(mu^2), where AU is arbitrary unit.

#### 20230413:
- **runagar.py**: Use `python runagar.py` to run analysis for all .nd2 files in `../1_data` folder (a test `1_data` folder is provided). Several folders will be generated, including `3_analysis`, `4_debug`, `5_param`.
- **Agarpad analysis.ipynb**: Including functions for generating histogram for fluorescent disctributions. 

#### 20240209:

-**runagar.py**: Bug fix. Sequentially generate image for all colors.