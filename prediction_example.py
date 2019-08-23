
import matplotlib.pyplot as plt
import matplotlib.image as Image
import numpy as np
import os

pair_num = 5

gallery_path = "./VRIC/gallery_images/"
probe_path = "./VRIC/probe_images/"
correct_pre = np.array([np.array(query_fids)[query_pids==pre_pids],pre_fids[query_pids==pre_pids]]).T
incorrect_pre = np.array([np.array(query_fids)[query_pids!=pre_pids],pre_fids[query_pids!=pre_pids]]).T


fig, ax = plt.subplots(pair_num,2)
axes = ax.ravel()
start = 10
count = 0
for i in range(start , start + pair_num):
    image = plt.imread(os.path.join(probe_path, correct_pre[i][0]))
    axes[count * 2].imshow(image)
    axes[count * 2].set_title(correct_pre[i][0])
    axes[count * 2].axis('off')
    image = plt.imread(os.path.join(gallery_path, correct_pre[i][1]))
    axes[count * 2 + 1].imshow(image)
    axes[count * 2 + 1].set_title(correct_pre[i][1])
    axes[count * 2 + 1].axis('off')
    count += 1
fig.tight_layout()

fig2, ax = plt.subplots(pair_num,2)
axes = ax.ravel()
count = 0
for i in range(start, start + pair_num):
    image = plt.imread(os.path.join(probe_path, incorrect_pre[i][0]))
    axes[count * 2].imshow(image)
    axes[count * 2].set_title(incorrect_pre[i][0])
    axes[count * 2].axis('off')
    image = plt.imread(os.path.join(gallery_path, incorrect_pre[i][1]))
    axes[count * 2 + 1].imshow(image)
    axes[count * 2 + 1].set_title(incorrect_pre[i][1])
    axes[count * 2 + 1].axis('off')
    count += 1
fig2.tight_layout()
