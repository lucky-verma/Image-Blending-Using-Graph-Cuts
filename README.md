# DAA_Project_1_Spring22

>  **MK97893, Lucky Verma**

## Installation

```
>>> pip install -r requirements.txt
>>> python setup.py install 
```

## Pymaxflow Usage

The method `add_grid_nodes` adds multiple nodes and returns their indices in a convenient n-dimensional array with the given shape; `add_grid_edges` adds edges to the grid with a given neighborhood structure (4-connected by default); and `add_grid_tedges` sets the capacities of the terminal edges for multiple nodes.
    
```python
import pymaxflow as mf

# Create a grid of nodes
nodes = mf.add_grid_nodes(shape=(10, 10))

# Add edges to the grid
mf.add_grid_edges(nodes, neighborhood=mf.neighborhoods.four_connected)

# Set the capacities of the terminal edges
mf.add_grid_tedges(nodes, capacities=[1, 1, 1, 1])

# Set the demands of the nodes
mf.set_demands(nodes, demands=[1, 1, 1, 1])
```

## Usage
- Testing images can be found in the `test` folder.

``` python
image_dir = r"{}/test".format(os.getcwd())
```

## TODO

- [x] cut of the graph into subgraphs and merge them into one. :shipit:
- [x] Get the segments of the nodes in the grid.
- [ ] couple of intermediate outputs
- [ ] Image showing clear overlapped region # Important
- [x] Min cut over images # Important
- [ ] Try **setuptools** to package the project.
