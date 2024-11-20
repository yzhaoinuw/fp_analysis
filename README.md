### Installation
1. Download "sleep_scoring_app_v{x}" (in which the {x} in the suffix "vx" denotes the current version) and unzip it as needed. Note if you unzip it to the same location where the zip file is, you may end up with a layered folder, ie., a "sleep_scoring_app_vx" inside a "sleep_scoring_app_vx". If this is the case , you need to peel it and move the inner "sleep_scoring_app_vx" somewhere else and delete the outer "sleep_scoring_app_vx". Please make sure you check this.  
2. Examine the content of the unzipped "sleep_scoring_app_vx". It should contain three folders: 1) _internal, 2) app_src, 3) models, and a .exe file called run_app.exe.

### Usage 
1. Double click "run_app.exe" to open the app. After you click "run_app.exe", it will open a page in your web browser. You don't need internet connection to run the app. The app only uses the web browser as the interface.
2. After the app opens, you will see two buttons: "Generate prediction" and "Visualize a recording". If you want to generate predicted sleep scores using the deep learning model, click "Generate prediction", then you will see the "Click here to select File" button. After you select a file, the app will briefly validate the file selected and then run the model in the background to generate predictions. You can check the progress bar for this process in the Terminal. 
3. When the prediction is completed, the app automatically shows the visualization of the prediction, which is the same interface you will see if you clicked "Visualize a recording". In this interface, you can interact with this visualization, including zooming and panning. You can also correct the model's prediction by selecting the Box Select tool to the upper right of the visualization. Once you select Box Select, you can then draw a region in EEG, EMG, or the NE plot and then press 1, 2, or 3 or the keyboard to make the annotations. 
4. If you draw a Select Box in the EEG plot and if the box spans less than 300 seconds, you will see the spectral density estimation (SDE) plot on top of the EEG plot. It shows you the density estimation (based on [Fast Fourier Transform](https://docs.scipy.org/doc/scipy/tutorial/fft.html#d-discrete-fourier-transforms)) of frequencies up to 20 Hz in the box you draw.
4. After you are done annotating the sleep scores, click the "Save Annotations" button located on the bottom left. Your annotations will be saved in the same prediction file.
5. You can also visualize a recording with or without running prediction. You will still be able to make annotation or manually label sleep scores for a recording from scartch, and save your annotations at any point.

#### Notes:
1. Please do not move the files or the folders around in the "sleep_scoring_app_vx" folder, it may break the app.
2. If you don't have a preprocessed sleep data file yet and you want to test run the app, you can use a sample preprocessed file from the "\sleep_data_labeled\preprocessed_data" folder for a demo. Look for it in the same Onedrive folder where you download the app.
3. If you are not presented with a file finder window that lets you name the file to be saved after you clicked "Save Annotations", it's likely that the browser you use automatically downloads it to a Download folder. You can check or change the setting of your browser to make sure you save the mat file to a location you want.

#### Demo
https://github.com/yzhaoinuw/sleep_scoring/assets/22312388/1f83cd67-6952-47d0-9c42-34487f3bb047

#### FAQ
v0.6.0 (added 11/28/2023)
1. **Why do the EEG, EMG, and NE signals not recover after I click *Reset*?**
A: When clicking Reset Axes after a very close up zoom-in, you may see that the signals don't recover or take a long time to recover. When this happens, simply slightly adjust the zoom level by scrolling on your mouse, after a split second, the signals should recover.

2. **Why does the sdreamer model seem to leave the last couple of seconds of my data unscored?**
A: When using the sdreamer model, you may notice that the last couple of seconds of signals unscored. But it should be no more than 64 seconds (if it is, please report to us). This is because the sdreamer model processes the signals based on 64-second sections. If the total duration of the signals cannot divide 64 evenly, there will be a leftover that's unscored. We plan to fix this issue in the next model.

3. **When zooming in or out, the upsampling of the signals is very slow to catch up.**
A: if you find the upsampling of the signals slow to catch up as you zoom in or out, try to slow down your zooming and see if that helps.

4. **Why does my annotation not work?**
A: First, check that you have selected **Box Select** above the graph on the right. Sometimes it maybe hidden. But as soon as you hover your cursor over there, it will appear. Second, make sure that the key you pressed is among the accepted keys. For example, if you are visualizing 3-class prediction results, then only **1**, **2**, and **3** are available for annotation. If it's a 4-class prediction result, then you would have the additioinal key, **4**, available. Any other keypress is ignored. 


#### Build from Source (Run using Anaconda)
There are three preparation processes that you need to do beforing using the app with Anaconda.

1. Install Miniconda, a minimal install of Anaconda. Follow the instrcutions here: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

2. Sign up for a GitHub account if you haven't. This is needed because, our sleep scoring app repository hosted on GitHub is a private repo because a model the app uses is not yet published. And as a private repo, you must have a GitHub account that you can share with me. Once I add you as a collaborator, you can download it and get updates from it. Follow the instructions here to sign up: https://docs.github.com/en/get-started/signing-up-for-github/signing-up-for-a-new-github-account. After you sign up, please send me your GitHub account name and I will add you to our private repo.

3. Get Git if you haven't. You need it to download the repo and getting updates. Follow the instructions here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

#### Set up the environment
After you have done the prep work above, open you anaconda terminal or anaconda power shell prompt, and then replicate the conda environment we use to develop the app by typing
```bash
conda env create -f environment.yml
```
You only need to do this **once**, but it is important that you do this step successfully to ensure a smooth run of the app.

#### Activate  the environment
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

