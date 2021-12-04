import matplotlib.pyplot as plt
from matplotlib import gridspec

from IPython.display import clear_output


def plot_batch_with_results(batch_imgs, batch_gts, results, outdir = None):
    """
    Plots images, GT segmentations and generated segmentations
    Input:
        batch_imgs - tensor of size (batch_size, 3, h, w)
        batch_gts - tensor of size (batch_size, 1, h, w)
        results - tensor of size (batch_size, 1, h, w)
        outdir - where the results should be saved
    """
    batch_size = batch_imgs.shape[0]
    rows = 3

    fig = plt.figure(figsize=(batch_size * 5, rows * 4))
    gs = gridspec.GridSpec(rows, batch_size, wspace=0.0, hspace=0.0)
    
    for img_num in range(batch_size):
        ax = plt.subplot(gs[0,img_num])
        ax.axis('off')
        ax.imshow(batch_imgs[img_num].permute(1,2,0))
        
        ax = plt.subplot(gs[1,img_num])
        ax.axis('off')
        ax.imshow(batch_gts[img_num].permute(1,2,0))

        ax = plt.subplot(gs[2,img_num])
        ax.axis('off')
        ax.imshow(results[img_num].permute(1,2,0))
        
    fig.tight_layout()
    clear_output()
    plt.show()
