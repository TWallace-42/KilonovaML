# KilonovaML
The final, and generally clearer repository for the code necessary to generate kilonova lightcurves using normalised flow machine learning. If there are any issues within the code you should contact me at thomas42wallace@gmail.com

# Workflow
There are a number of steps I took with each programme acting like a piece in a pipeline that starts with some inputs and a model and eventually outputs a normalised flow machine learning model

## Creating data
The ``DU17_Model.py'' file is the one that created data for me. If you are looking to train on new data this is the programme to swap out (i.e. make new data). In general I found that due to the large amount of data being created it was best to multithread to try and reduce the time taken but still this proved a long process. It might be worth looking into smaller batch data creation over a longer time rather than this programme that needs to be run in one session

## Preparing Data
