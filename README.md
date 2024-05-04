# EDUset ONE vision

The repository was created as part of a master's thesis at the Brno University of Technology. The main motivation for the project was the deployment of a computer vision application in the robotic cell EDUset ONE, located at Intemac Solution s.r.o. Within the thesis, three leading detection frameworks, R-CNN, SSD, and YOLO, were deployed. Subsequently, the frameworks were compared based on their own criteria. The repository can serve as a starting point for implementing these frameworks on custom datasets.

## Custom dataset of semi-finished goods

The custom dataset used to train the detection neural networks is available at the following two links:

1. [Roboflow](https://universe.roboflow.com/jan-zmrzly/eduset-one-dataset)
2. [Kaggle](https://www.kaggle.com/datasets/janzmrzly/eduset-one-dataset)


## Samples

In the [_samples_](https://github.com/JanZmrzly/eduset-vision/tree/main/samples) folder, there are examples of Jupyter notebooks that were used to work with detection frameworks. These examples demonstrate how it is possible to deploy each of the detection frameworks on a custom dataset. The background for the examples is in the [_eduset_](https://github.com/JanZmrzly/eduset-vision/tree/main/eduset) subfolder. This folder can serve as the default package for subsequent work.

# Main