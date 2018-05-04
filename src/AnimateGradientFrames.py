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
    # Get a list of subdirectories one level down in the parent source_dir:
    subdirs = next(os.walk(source_dir))[1]
    images = []
    subdirs = [e.name for e in os.scandir(source_dir) if e.is_dir()]
    subdir_anim_paths = [os.path.join(source_dir, subdir + '/anim') for subdir in subdirs]
    # Walk through every directory in PyTorchCNNVisualizations that has an anim subfolder:
    for path in subdir_anim_paths:
        # if the path doesn't exist yet, its our first time making it so create it:
        if not os.path.isdir(path):
            os.mkdir(path)
        # Now to find out which class this path pertains to:
        # for class_dir in subdirs:
        #     if class_dir in path:


        # Walk anim folder contents
        print('path', path)
        print('walking:')
        filenames = next(os.walk(path))[2]
        print('filenames:', filenames)

        for file in filenames:
            image = cv2.imread(os.path.join(path, file))
            images.append(image)
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
