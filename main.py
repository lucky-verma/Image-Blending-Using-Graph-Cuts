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


def compute_edge_weights(src, sink):
    """
    Computes edge weights based on matching quality cost.
    :param src: image to be blended (foreground)
    :param sink: background image
    """
    edge_weights = np.zeros((src.shape[0], src.shape[1], 2))

    src_left_shifted = np.roll(src, -1, axis=1)
    sink_left_shifted = np.roll(sink, -1, axis=1)
    src_up_shifted = np.roll(src, -1, axis=0)
    sink_up_shifted = np.roll(sink, -1, axis=0)
    eps = 1e-10

    weight = np.sum(np.square(src - sink, dtype=np.float) +
                    np.square(src_left_shifted - sink_left_shifted, 
                    dtype=np.float),
                    axis=2)
    norm_factor = np.sum(np.square(src - src_left_shifted, dtype=np.float) +
                         np.square(sink - sink_left_shifted, 
                         dtype=np.float),
                         axis=2)
    edge_weights[:, :, 0] = weight / (norm_factor + eps)

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


# Args parser
parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='image_dir', required=True, help='Image directory')
args = parser.parse_args()
image_dir = args.image_dir
src = cv2.imread(os.path.join(image_dir, 'src.jpg'))
sink = cv2.imread(os.path.join(image_dir, 'target.jpg'))
mask = cv2.imread(os.path.join(image_dir, 'mask.png'))

assert (src.shape == sink.shape), f"Source and sink dimensions must be the same: {str(src.shape)} != {str(sink.shape)}"

# Create the graph
graph = maxflow.Graph[float]()
node_ids = graph.add_grid_nodes((src.shape[0], src.shape[1]))

# Add edges
edge_weights = compute_edge_weights(src, sink)
patch_height = src.shape[0]
patch_width = src.shape[1]
for row_idx in range(patch_height):
    for col_idx in range(patch_width):
        if col_idx + 1 < patch_width:
            weight = edge_weights[row_idx, col_idx, 0]
            graph.add_edge(node_ids[row_idx][col_idx],
                        node_ids[row_idx][col_idx + 1],
                        weight,
                        weight)
        if row_idx + 1 < patch_height:
            weight = edge_weights[row_idx, col_idx, 1]
            graph.add_edge(node_ids[row_idx][col_idx],
                        node_ids[row_idx + 1][col_idx],
                        weight,
                        weight)
        if np.array_equal(mask[row_idx, col_idx, :], [0, 255, 255]):
            graph.add_tedge(node_ids[row_idx][col_idx], 0, np.inf)
        elif np.array_equal(mask[row_idx, col_idx, :], [255, 128, 0]):
            graph.add_tedge(node_ids[row_idx][col_idx], np.inf, 0)


# Intermediary results

#####
# Part 2.a
#####
path = os.path.join(image_dir, '2.a.Adj_matrix.txt')
with open(path, 'w') as f:
    for line in graph.get_nx_graph().adjacency():
        f.write(str(line) + '\n')


#####
# Part 2.a
#####

mask = cv2.imread(os.path.join(image_dir, 'mask.png'))
image = mask
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([22, 93, 0], dtype="uint8")
upper = np.array([45, 255, 255], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)

retval, thresh_crop = cv2.threshold(mask, thresh=254, maxval=255, type=cv2.THRESH_BINARY)
img_canny = cv2.Canny(thresh_crop, 10, 10)
img_dilate = cv2.dilate(img_canny, None, iterations=1)
img_erode = cv2.erode(img_dilate, None, iterations=1)

mask = np.full(thresh_crop.shape, 255, "uint8")
contours, hierarchies = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    cv2.drawContours(mask, [cnt], -1, 0, -1)

new_mask = cv2.cvtColor(cv2.bitwise_not(mask) , cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

src = cv2.imread(os.path.join(image_dir, 'src.jpg'))
sink = cv2.imread(os.path.join(image_dir, 'target.jpg'))
# Remove background using bitwise-and operation
result = cv2.bitwise_and(src, src, mask=thresh)
result[thresh==0] = [255,255,255]
_, mask = cv2.threshold(gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)
im_thresh_gray = cv2.bitwise_and(gray, mask)
res = cv2.bitwise_and(src,src,mask = mask)
cropped_src = cv2.cvtColor(res , cv2.COLOR_BGR2RGB)
dst = cv2.addWeighted(sink, 1.0, cropped_src, 1.0, 0)
cv2.imwrite('{}/2.a.png'.format(image_dir), dst)


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
resized_image = cv2.resize(img, src.shape[:2][::-1]) 
img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
plt.imshow(edges)
cv2.imwrite('{}/2.b.png'.format(image_dir), edges)
path = os.path.join(image_dir, '2.b.pixels.txt')
with open(path, 'w') as g:
    for line in edges:
        g.write(str(line) + '\n')

sys.exit()

