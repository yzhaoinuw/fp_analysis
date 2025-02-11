## Installation
1. Download "sleep_scoring_app_vx" (in which the x in the suffix "vx" denotes the current version) and unzip it as needed. Note if you unzip it to the same location where the zip file is, you may end up with a nested folder, ie., a "sleep_scoring_app_vx" inside a "sleep_scoring_app_vx". If this is the case , you need to peel it and move the inner "sleep_scoring_app_vx" somewhere else and delete the outer "sleep_scoring_app_vx". Please make sure you check this.  
2. Examine the content of the unzipped "sleep_scoring_app_vx". It should contain three folders, 1) **_internal**, 2) **app_src**, 3) **models**, and a .exe file called **run_app.exe**.


## Usage 
1. Double click "run_app.exe" to open the app. After you click "run_app.exe", it will open a page in your web browser. You don't need internet connection to run the app. The app only uses the web browser as the interface.
2. After the app opens, you will see two buttons: "Generate prediction" and "Visualize a recording". If you want to generate predicted sleep scores using the deep learning model, click "Generate prediction", then you will see the "Click here to select File" button. After you select a file, the app will briefly validate the file selected and then run the model in the background to generate predictions. You can check the progress bar for this process in Command Line Window. 
3. When the prediction is completed, the app automatically shows the visualization of the prediction, which is the same interface you will see if you had clicked "Visualize a recording" in Step 2. In this interface, you can interact with this visualization, including zooming and panning. You can also correct the model's prediction by selecting the Box Select tool to the upper right of the visualization or pressing "m" on the keyboard to switch to annotation mode. You can then draw a rectangle in EEG, EMG, or the NE plot and then press 1, 2, or 3 on the keyboard to make the annotations. To switch back to the panning mode, click Pan tool in the upper right or pressing "m" again.
4. A [spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.spectrogram.html#scipy.signal.ShortTimeFFT.spectrogram) of EEG of frequencies up to 20 Hz is calculated at the beginning of visualization. To view it, click on the small arrow "Show/Hide Spectrogram" above the signal figure. Click it again to fold it.
5. After you are done annotating the sleep scores, click the "Save Annotations" button located on the bottom left. Your annotations will be saved in the same prediction file.
6. You can also visualize a recording with or without running prediction. You will still be able to make annotations or manually label sleep scores for a recording from scartch, and save your annotations at any point.

### Video Demo
To visualize a recording

https://github.com/user-attachments/assets/07009c58-2aff-4fe1-84c8-472346718b4d

To generate a sleep score prediction

https://github.com/user-attachments/assets/8f826c2c-926c-48f5-b779-62485f443660


### Note:
In step 5, if you are not presented with a file finder window that lets you name the file to be saved after you clicked "Save Annotations", it's likely that the browser you use automatically downloads it to a Download folder. You can check or change the setting of your browser to make sure you save the mat file to a location you want.


## Input File 
The input files to the app must be .mat (matlab) files, and contain the following fields.
### Required Fields
| Field Name        | Data Type      |
| ------------------|----------------|
| **eeg**           | 1 x *N* single |
| **eeg_frequency** | double         |
| **emg**           | 1 x *N* single |

### Optional Fields
| Field Name    | Data Type      |
| --------------|----------------|
| ne            | 1 x *M* single |
| ne_frequency  | double         |
| sleep_scores  | single         |
| start_time    | uint32         |
 

## Build from Source (Run using Anaconda)
There are three preparation processes that you need to do beforing using the app with Anaconda.

1. Install Miniconda, a minimal install of Anaconda. Follow the instrcutions here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

2. Sign up for a GitHub account if you haven't. This is needed because, our sleep scoring app repository hosted on GitHub is a private repo because a model the app uses is not yet published. And as a private repo, you must have a GitHub account that you can share with me. Once I add you as a collaborator, you can download it and get updates from it. Follow the instructions here to sign up: https://docs.github.com/en/get-started/signing-up-for-github/signing-up-for-a-new-github-account. After you sign up, please send me your GitHub account name and I will add you to our private repo.

3. Get Git if you haven't. You need it to download the repo and getting updates. Follow the instructions here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

### Set up the environment
After you have done the prep work above, open you anaconda terminal or anaconda power shell prompt, and then replicate the conda environment we use to develop the app by typing
```bash
conda env create -f environment.yml
```
You only need to do this **once**, but it is important that you do this step successfully to ensure a smooth run of the app.

### Activate  the environment
Next, activate the enviroment called sleep_scoring_dist, by typing
```bash
conda activate sleep_scoring_dist
```
In the future before you run the app, you need to activate this environment.

#### download the source code for the app
```bash
git clone https://github.com/yzhaoinuw/sleep_scoring.git
```
At whatever directory you run this command will download the source code there. Of course, you can place the source code folder anywhere you like afterwards. Then, use the command *cd*, which stands for change directory, in your command line to change to where you place the sleep_scoring folder. 

#### Run the app
Last step, type
```bash
python app.py
```
to run the app.

#### Update the app
When there's an update announced, it's starightforward to get the update from source. Have the environment activated, cd to the source code folder, then type
```bash
git pull origin dev
```

