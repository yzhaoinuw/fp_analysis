## Installation
1. Go to the [fp_analysis folder](https://uofr-my.sharepoint.com/:f:/r/personal/yzhao38_ur_rochester_edu/Documents/fp_analysis_project?csf=1&web=1&e=JIeIs6) on Onedrive. Contact Yue if you can't access it.
2. Download **fp_analysis_app_vx.zip** (the "**x**" in the suffix "**vx**" denotes the current version). Note that if you unzip it to the same location where the zip file is, you may end up with a nested folder, ie., a "fp_analysis_app_vx" inside a "fp_analysis_app_vx". If this is the case , it may be better to peel it and move the inner "fp_analysis_app_vx" somewhere else and delete the outer "fp_analysis_app_vx".  
2. Examine the content of the unzipped "fp_analysis_app_vx". It should contain three folders and an .exe file: 1) **_internal**, 2) **fp_analysis_app**, and 3) **run_fp_analysis_app.exe**.


## Features And Usage 
To open the app, double click "run_fp_analysis_app.exe" and it will open the app's home page in a tab in your web browser. You don't need internet connection to run the app. The app only uses the web browser as the interface.

### Visualization
Click "**Click here to select File**". After you select a file, the app will briefly validate the file selected and then show you the figures of the the signals. Once the figures are shown, you can navigate and zoom in or out on the figures.

#### Zooming and Navigation
**Zooming** is done by scrolling your mouse around the center of interest. **Navigation** can be done using the left/right arrow key or just dragging the figure. To zoom or navigate along X-Axis only, move your cursor to the last figure and then operate. To zoom in or navigate along Y-Axis only, move your cursor to the left edge of the figure of interest and operate. 


### Known Issues
1. When adding a second or removing a second signal, clicking Show Results may not update the results even when the analyses are done (indicated by the Show Results button becomes available again), when this happens, switch to any other tab. That should help update the results. Fix to this issue is in progress.


## Input File 
The input files to the app must be .mat (matlab) files preprocessed using [sleep_data_preprocessing_app](https://github.com/yzhaoinuw/sleep_data_preprocessing_app) (although called sleep data, it also works with fp data), and contain the following fields. 
### Required Fields
| Field Name            | Data Type      |
| ----------------------|----------------|
| **_fp_signal_names_** | *N* x 1 string |
| **fp_frequency**      | float          |
| **_signal_A_**        | 1 x *N* single |
|  ...                  | 1 x *N* single | 

**Explanations**
1. **_signal_A_**  is just an example name for a fp signal. You can name it by its real signal name such as NE, but it must be listed as a name in **_fp_signal_names_**, in order to be visualized by the app.