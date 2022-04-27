# Image blending using GraphCuts

>  **MK97893, Lucky Verma**

## Setup Environment

Follow: [Download and install Python](https://www.python.org/downloads/release/python-380/)

Use Version: python3.8

# Installation

```
>>> pip3 install -r reqs.txt
```
# Project Description

The code is based on Kwatra et alpaper's "Graphcut Textures: Image and Video Synthesis Using Graph Cuts." There is also a graphcut algorithm built here, which takes own photos and applies the algorithm to them.

# Usage

There should be two images in the image directory: `src.jpg` and `target.jpg`. It is recommended that you resize your images to a low resolution, so that the blending process takes as little time as possible. The `test` directory contains a few examples.

> The resulting image will be saved in the same folder as the original.

The first intermediate result is shown in the files `2.a.Adj_matrix.txt` and intermediate `2.a.png`, which contains an image of two overlapping patches and illustrates where the overlap is and where it is not. A text file holding your graph's adjacency matrix or list of adjacencies.

The second intermediate result is represented by the files image `2.b.png` and `2.b.pixels.txt`, which contain an image that clearly colors the pixels of the identified cut (seam) and a text file containing a vector of pixel numbers that create cuts.

Finally, there's result.png, which combines two images to create a new one.

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
- **main.py:** Script that runs the min-cut max-flow algorithm for image blending
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
