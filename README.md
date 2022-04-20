# Linear Path Planner for Guided Grazing

Introducing the Linear Path Planner. Previous open source work at Avalanche to develop a physical fence robot for cattle and cows management to end over-grazing and yield up to 90% of hay excess.

![robot-overview](img/robot_1.png)
![robot-main-robot-detail](img/robot_2.png)
![robot-side-robot-detail](img/robot_3.png)

https://user-images.githubusercontent.com/18560386/164186889-5160a54d-dbb7-400c-9644-284d7deef19a.mp4

# How does it work?

[![Figma Presentation](img/figma_presentation.png)](https://www.figma.com/proto/dmTIbV9pCkTJNlniaIH6AF/path_planner_v0?node-id=1%3A2)
[Figma Presentation](https://www.figma.com/proto/dmTIbV9pCkTJNlniaIH6AF/path_planner_v0?node-id=1%3A2)

# Testing the Path Planner

![Parcel3](img/parcel_3.gif)

The entire workflow is detailed in `notebook/PathPlanner.ipynb`.
Let's run the planner on a new parcel.

## Creating the KLM File

First, head to Google Earth and create a mapping respecting the following requirements:
![GoogleEarthDemo](img/google-earth-demo.png)
  Create closed Polygons and name them: `Parcel {index} - Perimeter`.
  Naming is important because it will be use later by the GoogleEarthParser. Exemple:
  - Parcel 1 - Perimeter
  - Parcel 2 - Perimeter
  - Parcel 3 - Perimeter
  - Parcel 4 - Perimeter
2. Add the obstacles within your Polygon by drawing (you guess it) another set of Polygons
  They also must be named accordingly: `Parcel {index 1} - Obstacle {index 2}`. Exemple:
  - Parcel 1 - Obstacle 1
  - Parcel 1 - Obstacle 2
3. Add the Sweep axis by drawing a line. It will determine the orientation of our fences.
  Naming: `Parcel {index} - Sweep axis`. Note that you must assign only one sweep axis per field.
4. Finally, export the file as KML using the Google Earth UI.

## Setup the environment

- Requirements: python > 3.6

1. The next step is to install the Linear Path Planner locally and create the python environment.
Clone the repo, and before `cd` into it. Then run:
```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2. Afterwards, create a folder like `prod` or `test` and place your KML File in it.
3. Finally, open the `NewPathPlanner.ipynb` notebook. 
  Run `jupyter notebook notebook/NewPathPlanner.ipynb --log-level=CRITICAL&` to create a background thread and filter the log messages on terminal.
  You can also create a new notebook in the `notebook` directory and open it.

## Run the Planner!

Follow along if you have opened the NewPathPlanner, or use the high level API if you are creating a new notebook.
The main workflow is:

1. Parse the KML File into a parcel group instance. Set the working directory first in the class instantiation.

```python
from src.parsing.google_earth_parser import GoogleEarthParser

parser = GoogleEarthParser(workdir="../test")
parcel_group = parser.get_parcel_group("../test/MobileFence2.kml")
```

`parcel_group` is an instance of `ParcelGroup` with the following methods
`plot_parcels`






