# ML Animation Coloring Tool 


## Overview
This tool uses a machine learning-based approach to automatically color 2D animation cels.<br>
The data must first be preprocessed. The input consists of example fully colored cels with transparency and their corresponding line art layers with transparency.
Contours are extracted from each enclosed region, including regions that intersect the image boundaries. Shape features are then computed from these contours using standard shape feature extraction techniques. <sub><sup>[1]</sup></sub>
Color labels are assigned using a predefined color palette. The extracted features and corresponding labels are used to train a Random Forest classifier.
New frames are reconstructed using the trained classification model. Current results achieve a color prediction accuracy of 89.75%<br>

<br>


<sub><sup>[1] Mingqiang Yang, Kidiyo Kpalma, Joseph Ronsin. A Survey of Shape Feature Extraction Techniques. Peng-Yeng Yin. Pattern Recognition, IN-TECH, pp.43-90, 2008. hal-00446037</sup></sub>
