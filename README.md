# KilonovaML
The final, and generally clearer repository for the code necessary to generate kilonova lightcurves using normalised flow machine learning.

The Basic process is described in the chart below. Data must be presented to the Training programme whole and without nan values. I was able to create the DU17 model effectively using code within the gwemligthcurve package and a similair approach should work for other models in which case *hopeully* we can just change the data at the beginning and the rest of the process will follow as it has before. 

![alt text](https://github.com/2300431w/KilonovaML/blob/main/Flow%20Chart.png)

 #Instructions for running code

## Setting up the environment

Alongside the programmes and data provided in this github page you will need to create the following empty directories:

- DU17_training/
- Data_Cache/
- Model Evolution/
- Models/

An example of how the directory might look is shown below:

![alt text](https://github.com/2300431w/KilonovaML/blob/main/Folders.PNG)

## 1) Creating Data
Running DU17_Model.py will create data in the directory given in line 213 (DU17_Training/ by default). This process will take a long time if multiple threads are not used. To create smaller and more individual packets of data when run you can set ``N_threads = 1`` in line 196 and increase the multiplication factor in lines 199 to 202 (e.g. ``1*N_threads -> 16*N_threads``) 

*NB: the factor in lines 199-202 must not have a remainder when dividing the length of data as np.split splits data into even portions so if there are data points remaining there will be an error*

## 2) Data Combining
First you must move the data to the directory you wish to store it in (Data_Cache/ by default), This is where most data processing will be done and data can be stored. Data_combiner combines the data chunks outputed by DU17_Model.py found in the directory specified in line 6 (Data_Cache/ by default), if only one chunk is created this step is not necessary as it is just concatenating the multiple pandas dataframes into one. The result will then be outputed into the same directory that the data was found in as "combined.pkl".

*NB: If combined.pkl exists in the directory already it will be overwritten*

## 3) Removing `nan` values
To remove `nan` values we use the data_unnaner.py function. This removes `nan` values from the file specified in line 13 (Data_Cache/combined.pkl by default) by first reducing the data length by a factor `f` (defined in line 12) and setting `nan` values to their nearest real value (i.e. [`nan`, `nan`, 2,3, `nan`] becomes [2,2,2,3,3]. This will then produce a file in the same directory as specified in line 13 plus the additional term given in line 74 ({fname}+\_nannum.pkl) by default.

## 4) Training the AI
The AI will train from data in the directory specified in lines 45 and 71 (Whatever file is in DU17_Training/ by default). The unprocessed data (Data_Cache/combined.pkl by default but it is important to change if you move this file) is used to find scaling constants of the lightcurves. 
As the model trains it will output predicted graphs to "Model Evolution/" so that we can see the model evolve over time. When the model finishes it will save itself to "Models/". 

## 5) Using trained models
There are various programmes which use the created model. Model_user.py uses an individual model for a specific band while DU17_flow_Model uses four different models for each of the bands and is the final step of the AI generation. 

- Model_user.py creates random data using the model specified in the path in line 178 and compares it to the training data specified in line 71. You must also specify the band used in line 190 (g,r,i, or z).
- Model_user_time.py uses the model specified in line 97, gets scaling constants from the training data file specified in 112 as well as getting values of m1,m2,l1,l2 to generate data traditionally and compare the time taken for both
- DU17_Flow_Model.py uses the flow model of various different models specified in lines 34-37.

## Notes

Issues I had with this process that are worth looking out for:
- Creating data took a long time, multithreading reduced time but led to overheating issues with my computer. I have modified the code slightly to only make one module at a time with the segment created being a manual process (to change set ``N_threads  = [[actual # of threads]]`` and remove the ``*16`` in the following lines.)
- The Model failed on real world data. The leading theory as to why is that the training data I used has too specific a $\lambda_1$ = $\lambda_2$ so one solution might be to add noise. Though be careful, $m_1$ > $m_2$ and $\lambda_1$ < $\lambda_2$ . There is a rudimentary attempt at this in the comments of the DU17_Model.py file but I never got to test it. 
