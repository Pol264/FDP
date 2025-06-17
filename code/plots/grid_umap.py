from PIL import Image
import matplotlib.pyplot as plt

img1 = Image.open('umap_train_val_reduced_test_per_bin_8M_baseline_custom_colors.png')
img2 = Image.open('umap_train_val_reduced_test_per_bin_8M_our_model_custom_colors.png')
img3 = Image.open('umap_650M_baseline.png')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

images = [img1, img2, img3]
labels = ['(a)', '(b)', '(c)']

for ax, img, label in zip(axes, images, labels):
    ax.imshow(img)
    ax.set_title(label, loc='left', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('output_grid_8M_umap_8M_and_650M.png', format='png', dpi=300)  
plt.close()
