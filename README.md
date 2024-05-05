# EDUset ONE vision

The repository was created as part of a master's thesis at the Brno University of Technology. The main motivation for the project was the deployment of a computer vision application in the robotic cell EDUset ONE, located at Intemac Solution s.r.o. Within the thesis, three leading detection frameworks, R-CNN, SSD, and YOLO, were deployed. Subsequently, the frameworks were compared based on their own criteria. The repository can serve as a starting point for implementing these frameworks on custom datasets.

## Custom dataset of semi-finished goods

The custom dataset used to train the detection neural networks is available at the following two links:

1. [Roboflow](https://universe.roboflow.com/jan-zmrzly/eduset-one-dataset)
2. [Kaggle](https://www.kaggle.com/datasets/janzmrzly/eduset-one-dataset)


## Instalation

> [!WARNING]
> Installation instructions will be provided once the Python package is created

## Samples

In the [_samples_](https://github.com/JanZmrzly/eduset-vision/tree/main/samples) folder, there are examples of Jupyter notebooks that were used to work with detection frameworks. These examples demonstrate how it is possible to deploy each of the detection frameworks on a custom dataset. The background for the examples is in the [_eduset_](https://github.com/JanZmrzly/eduset-vision/tree/main/eduset) subfolder. This folder can serve as the default package for subsequent work.

## Main

In the [_main_](https://github.com/JanZmrzly/eduset-vision/tree/main/main) folder, there is a custom program for semi-finished goods detection. The main principle is that an API is created, which communicates with an industrial camera from BASLER. The API is invoked by an OPC UA client, which is connected to an OPC UA server running on a PLC from Siemens. The client and API can run independently of each other and can also be started independently using the following commands:

> Running the API on localhost:

```python
uvicorn.run(app, host="localhost", port=8000)
```

> Running the OPC UA client:

```python
url = "opc.tcp://192.168.0.10:4840"
    
api_endpoint = "http://localhost:8000/placement"

control_signal = {"ns": "http://EdusetONE", "i": 75}
placement = {"ns": "http://EdusetONE", "i": 19}

client = OPCUAClient(url=url)

try:
    await client.set_encryption() if client.encryption is True else None
    await client.run(control_signal, placement, time=2, api_endpoint=api_endpoint)
except KeyboardInterrupt:
    await client.disconnect()
```