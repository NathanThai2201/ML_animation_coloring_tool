# ML Animation Coloring Tool 


## Overview
This tool is a Machine Learning approach to automatically coloring 2d animation cells.<br>
First the data has to be preprocessed, input data require example full colored cells with transparency and their corresponding line layer with transparency.
Contours are extracted from every enclosed region including regions overlapping image borders.
Features are extracted from the contours according to common shape feature extraction techniques. <sub><sup>[1]</sup></sub><br>
labels are given by a palette of colors.<br> 
Features and Labels are trained with Random Forest Classifier.<br>
New frames are reconstructed with the classification model, current results color prediction accuracy at 89.28%<br>

> **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*<br>

<sub><sup>[1] Mingqiang Yang, Kidiyo Kpalma, Joseph Ronsin. A Survey of Shape Feature Extraction Techniques. Peng-Yeng Yin. Pattern Recognition, IN-TECH, pp.43-90, 2008. hal-00446037</sup></sub>