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
import tempfile
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files (x86)\\ffmpeg\\ffmpeg-4.0-win64-static\\bin\\ffmpeg.exe'
import matplotlib.animation as animation
import pylab as pl


def save_matplotlib_images(images):
    for clss, imgs in images.items():
        fig = plt.figure()
        fig.suptitle('class %s' % clss)
        for i, img in enumerate(imgs):
            im = plt.imshow(img, vmin=0, vmax=255)
            plt.title('epoch: %d' % i)
            plt.xticks([])
            plt.yticks([])
            plt.draw()
            write_path = '../results/PyTorchCNNVisualizations/' + str(clss) + '/plt'
            fname = write_path + '/vanilla_bp_color_plt_' + str(i) + '.png'
            if not os.path.isdir(write_path):
                os.mkdir(write_path)
            if not os.path.isfile(fname):
                plt.savefig(fname)


def animate(img_dict, clss, fps, write_path):
    """

    :param img_dict:
    :return:
    """
    if clss not in img_dict:
        print('The desired class to animate is not in the provided image dictionary!')
        exit(-1)
    # Get the images for the target class to animate:
    images = img_dict[clss]
    '''
    Animate the loaded images:
    '''
    fig = plt.figure()
    fig.suptitle('class %s' % clss)
    # plt.title('epoch: %d' % )
    im = plt.imshow(images[0], vmin=0, vmax=255)

    def updatefig(j):
        """
        updatefig: Updates the fig object with a new image from the images array.
        :param j: The index of the image in the images array to replace the current one with.
        :return [im]: TODO: Not sure what this is returning.
        """
        print('updatefig(j=%d)' % j)
        # Change the title to reflect the current epoch:
        plt.title('epoch: %d' % j)
        plt.draw()
        # set the data in the axesimage object
        im.set_array(images[j])
        # return the artists set
        return [im]
    # trigger animation:
    ani = animation.FuncAnimation(fig, updatefig, frames=len(images),
                                  interval=fps*1000, blit=True)
    # Save animation:
    # writer = animation.FFMpegWriter(fps=6, metadata=dict(title=clss), bitrate=None, extra_args=['-r', '1', '-pix_fmt', 'yuv420p'])
    # writer = animation.FFMpegWriter(metadata=dict(title=clss), extra_args=['-pix_fmt', 'yuv420p', '-framerate', '1', '-r', '30', '-c', 'libx264', '-f', 'gif'])
    # writer = animation.FFMpegWriter(fps=1)
    # writer = Writer(fps=fps, metadata=dict(clss=clss), bitrate=1800)
    # fname = str(clss) + '.mp4'
    # print(write_path)
    # ani.save(filename=write_path, writer=writer)
    # plt.show()


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
        for filename in filenames:
            file_ext = filename.split('.')[1]
            if file_ext != 'gif' and file_ext != 'mp4':
                if 'img_paths' not in img_meta[clss]:
                    img_meta[clss]['img_paths'] = []
                else:
                    img_meta[clss]['img_paths'].append(meta['anim_path'] + '/' + filename)
        # img_meta[clss]['img_paths'] = [meta['anim_path'] + '/' + fname for fname in filenames if fname.split('.')[1] is not '.gif' or fname
        #                                .split('.')[1] is not '.mp4']
        # print('filenames:', filenames)
    # print('img_meta', img_meta)
    """
    Load images from container directories:
    """
    images = {}
    for clss, meta in images_metadata.items():
        images[clss] = []
        for img_path in meta['img_paths']:
            image = cv2.imread(img_path)
            images[clss].append(image)
    """
    Save labeled images:
    """
    save_matplotlib_images(images)
    """
    Animate the loaded images:
    """
    animate(images, clss=281, fps=.5, write_path=os.path.join(source_dir, '281/281.mp4'))
    # img = None
    # for clss, meta in images_metadata.items():
    #     for img_path in meta['img_paths']:
    #         im = pl.imread(img_path)
    #         if img is None:
    #             img = pl.imshow(im)
    #         else:
    #             img.set_data(im)
    #         pl.pause(.5)
    #         pl.draw()


if __name__ == '__main__':
    results_dir = '../results/PyTorchCNNVisualizations/'
    main(source_dir=results_dir)
