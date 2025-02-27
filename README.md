## Installation
1. Go to the [sleep_scoring_project folder](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/ErxPdMtspCVDuXvfwtKK4rIBnIWP8SF5BkX-J2yD4MY11g) on Onedrive. Contact Yue if you can't access it.
2. Download **sleep_scoring_app_vx.zip** (the "**x**" in the suffix "**vx**" denotes the current version). Note that if you unzip it to the same location where the zip file is, you may end up with a nested folder, ie., a "sleep_scoring_app_vx" inside a "sleep_scoring_app_vx". If this is the case , it may be better peel it and move the inner "sleep_scoring_app_vx" somewhere else and delete the outer "sleep_scoring_app_vx".  
2. Examine the content of the unzipped "sleep_scoring_app_vx". It should contain three folders and an .exe file: 1) **_internal**, 2) **app_src**, 3) **models**, and 4) **run_app.exe**.


## Features And Usage 
To open the app, double click "run_app.exe" and it will open the app's home page in a tab in your web browser. You don't need internet connection to run the app. The app only uses the web browser as the interface.

### Automatic Sleep Scoring
Click **Generate prediction** at the top, then "**Click here to select File**". After you select a file, the app will briefly validate the file selected and then run the deep learning model in the background to generate sleep score predictions. You can check the progress bar for this process in Command Line Window. When the prediction is completed, the app will enter the visualization interface and show the visualization of the prediction. See the video below for a demo.

https://github.com/user-attachments/assets/8f826c2c-926c-48f5-b779-62485f443660

### Visualization
Click **Visualize a recording** at the top, then "**Click here to select File**". After you select a file, the app will briefly validate the file selected and then run the model in the background to generate predictions. Once the figures are shown, you can navigate and zoom in or out on the figures. See the video below for a demo.

#### Zooming and Navigation
**Zooming** is done by scrolling your mouse around the center of interest. **Navigation** can be done using the left/right arrow key or just dragging the figure. To zoom or navigate along X-Axis only, move your cursor to the last figure called "Prediction Confidence" and then operate. To zoom in or navigate along Y-Axis only, move your cursor to the left edge of the figure of interest and operate. 

#### Annotation
To **add or modify sleep scores**, press "m" on the keyboard to switch to annotation mode. You can then draw a rectangle in the EEG, EMG, or the NE plot and then press 1, 2, or 3 on the keyboard to make the annotations. To switch back to the drag mode, press "m" again. If you made a mistake, you can click the **Undo Annotation** button, which allows you to go back up to three moves. After you are done annotating the sleep scores, click the **Save Annotations** button located on the bottom left. Your annotations will be saved in the same prediction file.

#### Spectrogram
A **[spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.spectrogram.html#scipy.signal.ShortTimeFFT.spectrogram)** of EEG of frequencies up to 20 Hz is calculated at the beginning of visualization. To view it, click on the small arrow "Show/Hide Spectrogram" above the EEG figure. Click it again to fold it.

#### Check Video
When you draw a rectangle within the recording which spans less than **300 seconds** in duration, you should see a "Check Video" button appear just above the "Show/Hide Spectrogram" button. Click to open the video window. If you are checking the video for this recording for the first time, you will be prompted to upload the original .avi file. If the .avi file was found during [procrocessing](https://github.com/yzhaoinuw/preprocess_sleep_data/tree/dev), you will be provided with the path to help you locate the .avi file. 

https://github.com/user-attachments/assets/07009c58-2aff-4fe1-84c8-472346718b4d

### Note
1. If you are not presented with a file finder window that lets you name the file to be saved after you clicked "Save Annotations", it's likely that the browser you use automatically downloads it to a Download folder. You can check or change the setting of your browser to make sure you save the mat file to a location you want.
2. It may help the app run smoother if you close all other tabs in the same browser as the app.


## Input File 
The input files to the app must be .mat (matlab) files, and contain the following fields.
### Required Fields
| Field Name          | Data Type      |
| --------------------|----------------|
| **_eeg_**           | 1 x *N* single |
| **_eeg_frequency_** | double         |
| **_emg_**           | 1 x *N* single |

### Optional Fields
| Field Name         | Data Type      | 
| -------------------|----------------|
| *ne*               | 1 x *M* single | 
| *ne_frequency*     | double         |
| *sleep_scores*     | single         |
| *start_time*       | double         |
| *video_name*       | char           |
| *video_path*       | char           | 
| *video_start_time* | double         |

**Explanations**

 1. *start_time* is not *0* if the .mat file came from a longer recording (>12 hours) that was segmented into 12-hour or less bins.
 2. *video_path* is the .avi path found during preprocessing.
 3. *video_start_time* is the TTL pulse onset found on the EEG side (such as Viewpoint, Pinnacle).
 

## Build From Source (Run Using Anaconda)
There are two preparation processes that you need to do beforing using the app with Anaconda.

1. Install Miniconda, a minimal install of Anaconda. Follow the instrcutions here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

2. Get Git if you haven't. You need it to download the repo and getting updates. Follow the instructions here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

### Download Source Code
```bash
git clone https://github.com/yzhaoinuw/sleep_scoring.git
```
At whatever directory you run this command will download the source code there. You can place the source code folder anywhere you like afterwards. Then use the command *cd*, which stands for change directory, in your command line to change to where you place the sleep_scoring/ folder. 

### Set up the environment
After you have done the prep work above, open you anaconda terminal or anaconda power shell prompt, create an environment with Python 3.10
```bash
conda create -n sleep_scoring python=3.10
```
Then, activate the sleep_scoring environment by typing
```bash
conda activate sleep_scoring_dist
```
In the future, everytime before you run the app, make sure you activate this environment. Next, When you are in the sleep_scoring/ directory, install all the dependencies for the app. You only need to do it once ever.
```bash
pip install -r requirements.txt 
```


### Running The App
Last step, type
```bash
python app.py
```
to run the app.

### Updating the app
When there's an update announced, it's starightforward to get the update from source. Have the environment activated, cd to the source code folder, then type
```bash
git pull origin dev
```
