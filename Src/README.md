# Main Algorithms
There are multiple implementations based on the [this excel table](xeye_ai%20v1.2%20(1).xlsx).

[1. ver 1.0 : Ezzati Version](./0_Main_Algorithm_ver_1_0/empty_table-121.ipynb)

[2. ver 1.2](./0_Main_Algorithm_ver_1_2/Empty%20Table.ipynb) :

## Test

```bash
cd ./Main_Algorithm_ver_1_2
python Empty_Table.py
```

## Usage
```bash
from Src.Main_Algorithm_ver_1_2.Empty_Table import Execution
execution = Execution(x)
execution.Run()
```

The final version (ver 1.3) can be run as follow:

## Test

```bash
cd ./Main_Algorithm_ver_1_3
python Empty_Table.py
```

## Usage
```bash
from Src.Main_Algorithm_ver_1_3.Empty_Table import Execution
execution = Execution(x)
execution.Run()
```

## Input
```bash
x = {
    "camera": "rtsp://rtsp:Ashkan123@172.16.60.123/Streaming/Channels/101/",
    "table":
        {
            "512215":[125,45,320,550],
            "513125":[136,105,451,215],
            "512320" : [159-20, 286-6, 180+15, 306+40]
      }
}
```

## Output
```bash
{
  "camera":"Link or file",
  "table":
    {
      "512215": {
        "Working Time": "12:10:52",
        "Person is at the table without sitting": "5:12:15",
        "Chair is not detected": "3:12:00",
        "Person is not at the table":"2:12:01"
      },
      "513125": {
        "Working Time": "10:04:12",
        "Person is at the table without sitting": "3:35:00",
        "Chair is not detected": "3:12:00",
        "Person is not at the table":"2:12:01"
      }
    }
}
```

The final version (ver 2.3) can be run as follow:

## Test
```bash
cd ./Main_Algorithm_ver_2_3
python Empty_Table.py
```

## Usage
```bash
from Src.Main_Algorithm_ver_2_3.Empty_Table import Execution
execution = Execution(x)
execution.Run()
```

## Input
```bash
x = {
    "camera": "rtsp://rtsp:Ashkan123@172.16.60.123/Streaming/Channels/401/",
}
```

## Output
```bash
{
  "camera":"Link or file",
  "table":
    {
      "1": {
        "Working Time": "12:10:52",
        "Person is at the table without sitting": "5:12:15",
        "Chair is not detected": "3:12:00",
        "Person is not at the table":"2:12:01"
      },
      "2": {
        "Working Time": "10:04:12",
        "Person is at the table without sitting": "3:35:00",
        "Chair is not detected": "3:12:00",
        "Person is not at the table":"2:12:01"
      }
    }
}
```

# Getting Started
    - 1: IndoorObjectDetection 
        * https://github.com/d-duran/IndoorObjectDetection
        * Darknet/YOLOv2/v3/v4 weights needed
        * Indoor DB needed: the dataset contains {'exit': 545,'fireextinguisher': 1684, 'chair': 1662, 'clock': 280, 'trashbin': 228, 'printer': 81, 'screen': 115}

    - 2: Indoor Object Detection - Vote Net:
        * https://github.com/ShivamThukral/Indoor-Object-Detection
        * needs ubuntu OS
        * GazeBo models needed
        * Indoor 3D DB needed: https://data.nvision2.eecs.yorku.ca/3DGEMS/
        
    - 3: Indoor Object Detection:
        * https://github.com/Amin-Azar/Indoor-Object-Detection
        * Yolo - resnet - DenseNet
        * doesn't contain table
    
    - 4: YOLOv8 Indoor Objects Detection
        * Kaggle Source Code
        * Training Yolo8
        * DB: 
            ^ https://www.kaggle.com/datasets/dataclusterlabs/table-image-dataset-indoor-object-detection

            ^ https://www.kaggle.com/datasets/thepbordin/indoor-object-detection

            ^ https://www.kaggle.com/datasets/putrikhairunnisa/resnet50-indoor-object-detection

            ^ https://zenodo.org/record/2654485


# Build and Test
در هر پوشه یک فایل requirements.txt قرار داده شده است. علاوه بر آن در هر پوشه، به فایل های README مراجعه شود 