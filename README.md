# Image blending using GraphCuts

>  **MK97893, Lucky Verma**

## Setup Environment

Follow: [Download and install Python](https://www.python.org/downloads/release/python-380/)

Use Version: python3.8

# Installation

```
>>> pip3 install -r reqs.txt
```

# Usage

There should be two images in the image directory: `src.jpg` and `target.jpg`. It is recommended that you resize your images to a low resolution, so that the blending process takes as little time as possible. The `test` directory contains a few examples.

## Step 1:

Create mask of images using the following command:
 
```
>>> python3 mask.py -d {path to test images}
```

### Step 2:

Run graph cuts with the following command:

```
>>> python3 main.py -d {path to test images}
```
# Project Directories

- **test:** Directory containing a folder of images
- **mask.py:** Script that can be used to make mask for `src` & `sink` images
- **main.py:** Placeholder directory to store input files containing integers
- **req.txt:** Contains modules that were used to build this project 
- **resources:** Directory containing resource scripts and image data

# Outputs
![Screenshot_2](https://user-images.githubusercontent.com/63258138/165436798-f18f654c-0735-49d6-b844-8fe15bcd2ac3.png)

# References

- http://cs.brown.edu/courses/csci1950-g/results/proj3/dkendall/
- https://dl.acm.org/doi/abs/10.1016/j.imavis.2008.04.014
- https://github.com/ErictheSam/Graphcut/blob/master/mincut.py
- https://pmneila.github.io/PyMaxflow/maxflow.html
- https://github.com/niranjantdesai/image-blending-graphcuts
- https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
- https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
