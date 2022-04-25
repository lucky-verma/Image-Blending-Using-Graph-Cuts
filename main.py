import sys
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import maxflow
import argparse
from matplotlib.pyplot import figure
from numpy import save

image_dir = r"{}/test".format(os.getcwd())
src = cv2.imread(os.path.join(image_dir, 'src.jpg'))
sink = cv2.imread(os.path.join(image_dir, 'target.jpg'))
mask = cv2.imread(os.path.join(image_dir, 'mask.png'))

def compute_edge_weights(src, sink):
    """
    Computes edge weights based on matching quality cost.
    :param src: image to be blended (foreground)
    :param sink: background image
    """
    edge_weights = np.zeros((src.shape[0], src.shape[1], 2))

    # Create shifted versions of the matrices for vectorized operations.
    src_left_shifted = np.roll(src, -1, axis=1)
    sink_left_shifted = np.roll(sink, -1, axis=1)
    src_up_shifted = np.roll(src, -1, axis=0)
    sink_up_shifted = np.roll(sink, -1, axis=0)

    # Assign edge weights.
    # For numerical stability, avoid divide by 0.
    eps = 1e-10

    # Right neighbor.
    weight = np.sum(np.square(src - sink, dtype=np.float) +
                    np.square(src_left_shifted - sink_left_shifted, 
                    dtype=np.float),
                    axis=2)
    norm_factor = np.sum(np.square(src - src_left_shifted, dtype=np.float) +
                         np.square(sink - sink_left_shifted, 
                         dtype=np.float),
                         axis=2)
    edge_weights[:, :, 0] = weight / (norm_factor + eps)

    # Bottom neighbor.
    weight = np.sum(np.square(src - sink, dtype=np.float) +
                    np.square(src_up_shifted - sink_up_shifted,
                    dtype=np.float),
                    axis=2)
    norm_factor = np.sum(np.square(src - src_up_shifted, dtype=np.float) +
                         np.square(sink - sink_up_shifted, 
                         dtype=np.float),
                         axis=2)
    edge_weights[:, :, 1] = weight / (norm_factor + eps)
    
    return edge_weights

assert (src.shape == sink.shape), f"Source and sink dimensions must be the same: {str(src.shape)} != {str(sink.shape)}"



# Create the graph
graph = maxflow.Graph[float]()
# Add the nodes. node_ids has the identifiers of the nodes in the grid.
node_ids = graph.add_grid_nodes((src.shape[0], src.shape[1]))

edge_weights = compute_edge_weights(src, sink)

# Add non-terminal edges
patch_height = src.shape[0]
patch_width = src.shape[1]
for row_idx in range(patch_height):
    for col_idx in range(patch_width):
        # right neighbor
        if col_idx + 1 < patch_width:
            weight = edge_weights[row_idx, col_idx, 0]
            graph.add_edge(node_ids[row_idx][col_idx],
                        node_ids[row_idx][col_idx + 1],
                        weight,
                        weight)

        # bottom neighbor
        if row_idx + 1 < patch_height:
            weight = edge_weights[row_idx, col_idx, 1]
            graph.add_edge(node_ids[row_idx][col_idx],
                        node_ids[row_idx + 1][col_idx],
                        weight,
                        weight)

        # Add terminal edge capacities for the pixels constrained to
        # belong to the source/sink.
        if np.array_equal(mask[row_idx, col_idx, :], [0, 255, 255]):
            graph.add_tedge(node_ids[row_idx][col_idx], 0, np.inf)
        elif np.array_equal(mask[row_idx, col_idx, :], [255, 128, 0]):
            graph.add_tedge(node_ids[row_idx][col_idx], np.inf, 0)


# Compute max flow / min cut.
flow = graph.maxflow()
sgm = graph.get_grid_segments(node_ids)

#####
# Display the result 2.c
#####

sink[sgm] = src[sgm]
image = sink
rgb_img = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
figure(figsize=(10, 6), dpi=80)
plt.imshow(rgb_img)
plt.savefig(os.path.join(image_dir, '2.c.png'))

#####
# Display the result 2.a
#####

# Create the color map of the graph

#####
# Display the result 2.b
#####

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))# Show the result.
fig = plt.figure(frameon=False)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(img2, aspect='auto')
fig.savefig("{}/temp.png".format(image_dir))
img = cv2.imread("{}/temp.png".format(image_dir))
os.remove("{}/temp.png".format(image_dir))
# resize image
resized_image = cv2.resize(img, src.shape[:2][::-1]) 
# Convert to graycsale
img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
plt.imshow(edges)
# Display Canny Edge Detection Image
cv2.imwrite('{}/2.b.png'.format(image_dir), edges)
# save to npy file
save('{}/2.b.npy'.format(image_dir), edges)



# # Compute intermediate result
# canvas = np.zeros(src.shape, dtype=np.uint8)

# # Fill in the pixels constrained to belong to the src.
# src[sgm] = canvas[sgm]
# plt.savefig(os.path.join(image_dir, "result_src.png"))
# plt.show()

# # Fill in the pixels constrained to belong to the sink.
# sink[sgm] = canvas[sgm] 
# plt.savefig(os.path.join(image_dir, "result_sink.png"))
# plt.show()
sys.exit()

