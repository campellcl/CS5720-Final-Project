"""
AnimateGradientFrames.py
Goes through each of the results/PyTorchCNNVisualizations/ looking for an 'anim' folder. If the folder is present, this
    class will utilize matplotlib to create an animation out of all the gradient snapshots present.
"""

__author__ = "Chris Campell"
__created__ = "5/2/2018"

import os
import cv2
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main(source_dir):
    '''
    main:
    :param source_dir:
    :return:
    '''
    """
    Setup metadata container:
    """
    # Get a list of subdirectories one level down in the parent source_dir:
    subdirs = next(os.walk(source_dir))[1]
    # subdirs = [e.name for e in os.scandir(source_dir) if e.is_dir()]
    images_metadata = dict.fromkeys(int(clss) for clss in subdirs)
    # print('images_metadata:', images_metadata)
    # Add the anim/ subfolder path to every class label in the metadata:
    for subdir in subdirs:
        images_metadata[int(subdir)] = {'anim_path': os.path.join(source_dir, subdir + '/anim')}
    # Add the image path themselves to the metadata for each class:
    img_meta = images_metadata.copy()
    for clss, meta in images_metadata.items():
        filenames = next(os.walk(meta['anim_path']))[2]
        img_meta[clss]['img_paths'] = [meta['anim_path'] + '/' + fname for fname in filenames]
        # print('filenames:', filenames)
    print('img_meta', img_meta)
    """
    Load images from container directories:
    """
    images = []

    for clss, meta in images_metadata.items():
        for img_path in meta['img_paths']:
            image = cv2.imread(img_path)
            images.append(image)

    #     for file in filenames:
    #         image = cv2.imread(os.path.join(path, file))
    #         images.append(image)
            # Hide ticks on X and Y axis:
            # plt.xticks([]), plt.yticks([])
            # plt.imshow(image)
            # plt.show()
    # print(subdir_anim_paths)
    # print(subdirs)
    # plt_images = []
    # for img in images:
    #     im = plt.imshow(img, animated=True)
    #     plt_images.append([im])
    # fig = plt.figure()
    # anim = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=10000000000000000)
    # plt.show()
    fig = plt.figure()
    im = plt.imshow(images[0], vmin=0, vmax=255)
    # function to update figure

    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(images[j])
        # return the artists set
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=20,
                              interval=50, blit=True)
    plt.show()
    # for subdir, dirs, files in os.walk(source_dir):
    #     print('dirs', dirs)
    #     print('files', files)
    #     for dir in dirs:
    #         if os.path.isdir(os.path.join(source_dir, dir)):
    #             pass
            # if dir == 'anim':
            #     for file in os.walk(dir):
            #         print('file', file)


        # for file in files:
        #     relDir = os.path.relpath(subdir, source_dir)
        #     print(os.path.join(subdir, file))
        # print('subdir', subdir)


if __name__ == '__main__':
    results_dir = '../results/PyTorchCNNVisualizations/'
    if os.path.isdir(results_dir):
        print('dir exists')
    main(source_dir=results_dir)
