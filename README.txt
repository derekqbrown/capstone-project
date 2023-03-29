Image Classification using Machine Learning
By Derek Brown

Note: please see the "c964-proposal-DerekBrown" file for details

User Manual
Before running this software, you will need to have Python 3.10 installed on your system. Python 3.10 can be installed by going to this link and downloading either “Windows installer (64-bit)” or “Windows installer (32-bit).” This guide will also use the PyCharm IDE. Another IDE should also work, but instructions will not be provided for an alternative IDE. To install PyCharm, go here.

In addition to Python 3.10, the following libraries will need to be installed on the system if not already installed:  pandas, tensorflow, keras, matplotlib, numpy, sklearn, and opencv-python.  Users can install the required libraries in PyCharm  using the “Python Packages” option near the bottom. Follow this guide for more details. 

Once Python 3.10, PyCharm, and the libraries listed are installed, proceed to the steps below.


1.Download and unzip the folder. You can move the folder to your desktop for ease of access. 

2.Run PyCharm and create a new project. Use all the default settings. You should now see a main.py file.

3.Try running the new project to ensure that it was created properly. If the file doesn't print a message like "Hi, PyCharm" to the console, there is an issue with your setup. Follow this guide if this is the case.

4.If everything is working correctly, delete the main.py file (you should see the file listed on the left-hand side - right-click and select “Delete”). 

5.Right-click on the project folder and select “Open In” > “Explorer”

6.Drag and drop the folder “img-rec” from its current location into this project folder. You should now see the folder and its files where “main.py” formerly was.

7.Select “Run” and “Run ‘app’” at the top. See the image to the right.

● If “Run ‘app’” is not an available option, try right-clicking on the filename for “app.py” in PyCharm and finding the “Run ‘app’” option.

● If you have not properly installed the libraries listed above, you will get an error message when running the application. Follow this guide to install the libraries. 

8.You should now see the user interface, as pictured to the right.

9.To make a prediction on a batch of images, select the “Browse…” button and select the folder on the computer where the images are stored. 

● Alternatively, you can type the folder into the text field, replacing the text “Select a folder”.

10.Click the “Predict Results” button. You will see a confirmation notice. Select “OK” 

11.After the images have been processed, you should see a table displaying some results. 

● The table may not include all results for larger files; however, all the results will be found in the reports.csv file found in the resources folder (img-rec/resources/reports.csv). This file can be opened and viewed using Microsoft Excel and similar programs.

Additional features:
● Selecting the “Load Data” button will process all images found in the “data” folder and prepare them for retraining the current model. This should only be done if the data has been updated.
● Selecting the “Retrain the Model” button will allow the user to retrain the model. This should only be done if the current model needs to be updated with new data or is malfunctioning in some way, as this option will replace the current model with the newly trained model. 
● Selecting the “Accuracy of Model” button will display a line graph showing the level of accuracy (based on the training data).
● Selecting the “Results Summary” button will display a pie chart that shows the distribution of the images predicted. 
● The “Close” button closes the program.
● Pay attention to the status message at the bottom. If something goes wrong, it will indicate as much.
