import datasets
from matplotlib import pyplot as plt
import numpy as np
import torch

im1 = "./eer_lawn_unmasked.jpg"
im2 = "./eer_lawn_moremasked.jpg"

patch_size = 64
stride = 8
num_levels = 5
dataset = datasets.ModifiedPatchPatchDataset(im1, im2, patch_size, stride=stride)
rows = 1300 // stride
cols = 2200 // stride

print(f"patch image size: {cols}x{rows} = {cols * rows} patches")

# x, y = dataset[136]
# print(x.shape, y.shape)
img = np.zeros((1300 // stride, 2200 // stride))
for i in range(len(dataset)):
    og, modded = dataset[i]
    img[i // (2200 // stride), i % (2200 // stride)] = modded


plt.imshow(img)
plt.savefig("test.png")
