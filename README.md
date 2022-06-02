# KilonovaML
The final, and generally clearer repository for the code necessary to generate kilonova lightcurves using normalised flow machine learning.

The Basic process is described in the chart below. Data must be presented to the Training programme whole and without nan values. I was able to create the DU17 model effectively using code within the gwemligthcurve package and a similair approach should work for other models in which case *hopeully* we can just change the data at the beginning and the rest of the process will follow as it has before. 

![alt text](https://github.com/2300431w/KilonovaML/blob/main/Flow%20Chart.png)


Issues I had with this process that are worth looking out for:
- Creating data took a long time, multithreading reduced time but led to overheating issues with my computer. I have modified the code slightly to only make one module at a time with the segment created being a manual process (to change set ``N_threads  = [[actual # of threads]]`` and remove the ``*16`` in the following lines.)
- The Model failed on real world data. The leading theory as to why is that the training data I used has too specific a $\lambda_1$ = $\lambda_2$ so one solution might be to add noise. Though be careful, $m_1$ > $m_2$ and $l_1$ < $l_2$. There is a rudimentary attempt at this in the comments of the DU17_Model.py file but I never got to test it. 
