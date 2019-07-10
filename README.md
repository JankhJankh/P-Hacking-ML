# P-Hacking-ML
Repository for my SecTalks talk/exercises.

I don't own a lot of the data used in these exercises.
If the owners of any of the data feel this counts as a misuse of it, please shoot me a message and I'm happy to deconflict any issues.
This is all done to try to help people learn about the pitfalls of different Machine Learning (ML) algorithms in an attempt to improve overall ML security by getting people thinking about ML, and to make sure that testers are aware of the inner working of ML and not treating it as a black box when testing them for vulnerabilities.

For the exercises to work, make sure you have a 64 bit of python 3.7 installed, and then install the requirements with pip.
For the Kmeans challenge, you must unzip the image folders in the static directory.

Solutions:
For the solutions, you will need to rename the Kmeans solution folder to K-Means for it to be ran on the website. Each folder contains a mod.py for how that dataset was modified.
For CNN you can see my resultant image is trixi_frog.png, if you want to generate your own version you will have to delete the current trixi_mod.png and trixi_frog.png, and then make a copy of trixi.png called trixi_mod.png. The python file in that directory should now successfully create the adversarial noise version of the trixi.png called trixi_frog.png


Thanks for stopping by,

Jankh.
