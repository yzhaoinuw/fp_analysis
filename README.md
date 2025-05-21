## Installation
1. Go to the [Fiber Photometry Analysis folder](https://uofr-my.sharepoint.com/:f:/r/personal/yzhao38_ur_rochester_edu/Documents/Sleep%20Data%20Post-Analysis%20Project/Fiber%20Photometry%20Analysis?csf=1&web=1&e=Ud7UdT) on Onedrive. Contact Yue if you can't access it.
2. Download **fp_visualization_app_vx.zip** (the "**x**" in the suffix "**vx**" denotes the current version). Note that if you unzip it to the same location where the zip file is, you may end up with a nested folder, ie., a "fp_visualization_app_vx" inside a "fp_visualization_app_vx". If this is the case , it may be better to peel it and move the inner "fp_visualization_app_vx" somewhere else and delete the outer "fp_visualization_app_vx".  
2. Examine the content of the unzipped "fp_visualization_app_vx". It should contain three folders and an .exe file: 1) **_internal**, 2) **app_src**, and 3) **run_app.exe**.


## Features And Usage 
To open the app, double click "run_app.exe" and it will open the app's home page in a tab in your web browser. You don't need internet connection to run the app. The app only uses the web browser as the interface.

### Visualization
Click "**Click here to select File**". After you select a file, the app will briefly validate the file selected and then show you the figures of the the signals. Once the figures are shown, you can navigate and zoom in or out on the figures.

#### Zooming and Navigation
**Zooming** is done by scrolling your mouse around the center of interest. **Navigation** can be done using the left/right arrow key or just dragging the figure. To zoom or navigate along X-Axis only, move your cursor to the last figure and then operate. To zoom in or navigate along Y-Axis only, move your cursor to the left edge of the figure of interest and operate. 

#### Annotation
To **add or modify sleep scores**, press "m" on the keyboard to switch to annotation mode. You can then draw a rectangle in any of the signal plots and then press 1, 2, or 3 on the keyboard to make the annotations. To switch back to the drag mode, press "m" again. If you made a mistake, you can click the **Undo Annotation** button, which allows you to go back up to three moves. After you are done annotating the sleep scores, click the **Save Annotations** button located on the bottom left. Your annotations will be saved in the same prediction file.

#### Check Video
WHile in annotation mode, when you draw a rectangle within the recording which spans less than **300 seconds** in duration, you should see a "Check Video" button appear just above the top plot. Click to open the video window. If you are checking the video for this recording for the first time, you will be prompted to upload the original .avi file. If the .avi file was found during [procrocessing](https://github.com/yzhaoinuw/preprocess_sleep_data/tree/dev), you will be provided with the path to help you locate the .avi file. 

### Note
1. If you are not presented with a file finder window that lets you name the file to be saved after you clicked "Save Annotations", it's likely that the browser you use automatically downloads it to a Download folder. You can check or change the setting of your browser to make sure you save the mat file to a location you want.
2. It may help the app run smoother if you close all other tabs in the same browser as the app.
3. Although a different app, but you can watch the video demo for the [Sleep Scoring App](https://github.com/yzhaoinuw/sleep_scoring) for usage reference.


## Input File 
The input files to the app must be .mat (matlab) files, and contain the following fields.
### Required Fields
| Field Name            | Data Type      |
| ----------------------|----------------|
| **_fp_signal_names_** | *N* x 1 string |
| **_signal_A_**        | 1 x *N* single |
|  ...                  | 1 x *N* single | 

### Optional Fields
| Field Name         | Data Type      | 
| -------------------|----------------|
| *sleep_scores*     | single         |


**Explanations**
1. **_signal_A_**  is just an example name for a fp signal. You can name it by its real signal name such as NE, but it must be listed as a name in **_fp_signal_names_**, in order to be visualized by the app.
 