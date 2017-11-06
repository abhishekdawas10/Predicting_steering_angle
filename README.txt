===================================
README
Eeshaan Sharma - 2015CSB1011
Shivam Mittal  - 2015CSB1032
===================================
NOTE - Q1), Q2) and the competition part, all are implemented in MATLAB


Q1) Warm up Exercise

How To Run :

	1.) Navigate to the following directory - /code/Q1
	2.) In the Matlab window type - l31
	3.) Follow the input prompt to run on various values of number of epochs,
	    learning rate, number of neurons in hidden layer.

Q2.1) Predicting the Steering Angle

NOTE - Q2.1) requires the steering folder to be present in code folder  

How To Run :

	1.) Navigate to the following directory - /code/Q2_1
	2.) In the Matlab window type - Q2_1
	3.) Follow the input prompt to run on various values of number of epochs,
	    learning rate, mini batch size and dropout probability of different layers.

NOTE - The images present in steering folder will be read on first run of the code and then the
       input matrix X (1 X 1024) will be stored in Images.csv file.
	   
Q2.2) Competition part

The output for the test images is given in the text file test_output.txt

Note - For this part, we have extracted features from the images,
both for the training/validation images and the test images. 
Also, our model is an ensemble of 3 neural networks, so there
are 3 weight files corresponding to each neural network. 
Download all the files from the google drive folder :
https://drive.google.com/drive/u/0/folders/1mv5iiffdhE0Mdfi4fEhfyznOzR8CxvfL
and include these in the folder Q2_2.
Otherwise you'll have to wait a very very long time for 3 neural nets to train.

If you do not have the given files, and want to train the network again
then, steering folder and l3-test folder should be in the parent directory (code) of 
Q2_2.


Also note, if you want to test on some different images, then delete the features_test.csv 
file and put those images and text file in l3-test folder.


Running/compiling
------------------

    1.) Navigate to the following directory - /code/Q2_2
    2.) In the Matlab window type - main

