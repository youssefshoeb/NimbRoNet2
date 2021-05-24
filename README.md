# NimbRoNet2
This repository contains an implementation of the visual perception system of the soccer robot NimbRoNet2 in PyTorch

Original paper <https://arxiv.org/abs/1912.07405>

The repository also includes an improvement to the original model which decreases the number of parameters by almost double. More details can be found in the `Report.pdf`

## Files
 `main.ipynb` contains the main training loop

 `detection_evaluation.ipynb` evaluates the performance of the model on the test set and displays the results of the detection head

 `segmentation_evaluation.ipynb` evaluates the performance of the model on the test set and displays the results of the segmentation head

 `model.py` contains the source code defining the network architectures used in the project  

 `dataset.py` contains the source code to load and preprocess the data

 `utils.py` contains the source code helper functions used throughout the project

 `Report.pdf` a summary of the project and achieved results

 `requirements.txt` list of project dependencies

 