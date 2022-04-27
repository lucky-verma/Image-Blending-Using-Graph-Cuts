# DAA_Project_1_Spring22

>  **MK97893, Lucky Verma**

# Installation

```
>>> pip install -r reqs.txt
```

# Usage

There should be two images in the image directory: src.jpg and target.jpg. It is recommended that you resize your images to a low resolution, such as 640 x 480, so that the blending process takes as little time as possible. The test directory contains a few examples.

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


## TODO

- [x] Get the segments of the nodes in the grid.
- [x] couple of intermediate outputs.
- [x] Image showing clear overlapped region.
- [x] Min cut over images.
- [ ] Try **setuptools** to package the project.