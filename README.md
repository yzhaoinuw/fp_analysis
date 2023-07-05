## Standalone App Installation and Usage
1. Download the sleep_scoring_app.exe and the .mat files needed for demo from [](https://drive.google.com/drive/u/0/folders/13nNVXYcjN2yw_9LVrmXtooh5K9VDSyxJ).
2. Download the model weighteegxnexemg-[3. 5. 7.].h5 and put it in the same directory as the sleep_scoring_app.exe.
3. Double clicks the app.exe to open the app, which will open a page in a web browser.
4. Choose "Generate prediction" if you want to generate sleep score predictions on the data.mat or choose "Visualize existing prediction" if you want to visualize the predictions.
5. Click Select Files. If you chose "Generate prediction", then use the file finder that should pop up now to select the data.mat you downloaded together with this app. If you chose "Visualize existing prediction", then select results.mat.
6. The app should then either write a final_results.mat file if you chose "Generate prediction", or show the visualization in the same web page if you chose "Visualize existing prediction".