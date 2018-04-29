"""
visualizer.py
Driver file for multiple CNN visualizations using a variety of techniques.
"""
import argparse
import os
import shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms, utils
from src.misc_functions import get_params, convert_to_grayscale, save_gradient_images
from src.vanilla_backprop import VanillaBackprop

'''
Command Line Argument Parsing
'''
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
# Types of visualizations this program can produce:
visualization_names = ['vanilla_backprop']
parser = argparse.ArgumentParser(description='Visualizing CNN Gradients via Colored Standard Backpropagation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--viz', '-v', metavar='VIZ', default='vanilla_backprop', choices=visualization_names,
                    help='type of visualization to produce '
                         + ' | '.join(visualization_names)
                         + ' (default: vanilla_backprop')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# How many images will the data loader grab during one call to next(iter(data_loader)) or iter(data_loader).next()
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# Shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: False).
parser.add_argument('--shuffle', '-s', default=False, type=bool, metavar='SHUFFLE',
                    help='Reshuffle the data at every epoch? (default: False)')
# num_workers (int, optional): how many subprocesses to use for data loading.
#   0 means that the data will be loaded in the main process. (default: 4)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Args passed in to command line:
args = parser.parse_args()

'''
Global Variables:
'''
# Flag indicating if GPU should be used for training:
use_gpu = torch.cuda.is_available()


def get_data_transforms(dataset_is_present, input_load_size=256, receptive_field_size=224):
    """
    get_data_transforms: Gets the data transformation pipeline for each of the three input datasets.
    :param dataset_is_present: A dictionary of boolean flags (one for each dataset) indicating if the source directory
        is present on the OS. Flags for: train, val, test, viz.
    :param input_load_size: The dimensions (x, y) for which the input image is to be re-sized to.
    :param receptive_field_size: The dimensions (x, y) for which the input image is to cropped to (either a center or
        randomized crop depending on the defined pipeline).
    :return data_transforms: A dictionary housing sequential function calls that perform transformations to images in
        each of the datasets: train, val, and test.
    """
    data_transforms = {}
    for dataset_name, is_present in dataset_is_present.items():
        if dataset_name == 'train' and is_present:
            data_transforms[dataset_name] = transforms.Compose([
                transforms.RandomResizedCrop(receptive_field_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif dataset_name == 'val' and is_present:
            data_transforms[dataset_name] = transforms.Compose([
                transforms.Resize(input_load_size),
                transforms.CenterCrop(receptive_field_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif dataset_name == 'test' and is_present:
                data_transforms[dataset_name] = transforms.Compose([
                transforms.Resize(input_load_size),
                transforms.CenterCrop(receptive_field_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif dataset_name == 'viz' and is_present:
            data_transforms[dataset_name] = transforms.Compose([
                transforms.Resize(receptive_field_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    return data_transforms


def get_image_loaders(data_transforms):
    """
    get_image_loaders: Returns image loaders for all three datasets if present. This is an intermediary step to creating
        a torch.utils.data.DataLoader which handles asynchronous mini-batch loading.
    :param data_transforms: A dictionary housing sequential function calls that perform transformations to images in
        each of the datasets: train, val, and test.
    :return image_loaders: A dictionary containing references to the image loaders for each dataset.
    """
    image_loaders = {}
    for dataset_name, transformation_pipeline in data_transforms.items():
        if dataset_name is not None:
            if dataset_name == 'train':
                # Training set image loader:
                trainset_imgloader = datasets.ImageFolder(os.path.join(args.data, 'train'), data_transforms['train'])
                image_loaders[dataset_name] = trainset_imgloader
            elif dataset_name == 'val':
                # Validation set image loader:
                valset_imgloader = datasets.ImageFolder(os.path.join(args.data, 'val'), data_transforms['val'])
                image_loaders[dataset_name] = valset_imgloader
            elif dataset_name == 'test':
                # Testing set image loader:
                testset_imgloader = datasets.ImageFolder(os.path.join(args.data, 'test'), data_transforms['test'])
                image_loaders[dataset_name] = testset_imgloader
            elif dataset_name == 'viz':
                # Visualization set image loader:
                vizset_imgloader = datasets.ImageFolder(os.path.join(args.data, 'viz'), data_transforms['viz'])
                image_loaders['viz'] = vizset_imgloader
    return image_loaders


def get_data_loaders(image_loaders):
    """
    get_data_loaders: Returns a dict of data loader references, one for each dataset: train, val, test. The instances
        of torch.utils.data.DataLoader handles asynchronous loading of mini-batches.
    :param image_loaders: A dictionary containing references to the image loaders for each dataset.
    :return data_loaders: A dictionary of data loader instance references, one for each dataset: train, val, and test.
    """
    data_loaders = {}
    for dataset_name, image_loader in image_loaders.items():
        if dataset_name is not None:
            if dataset_name == 'train':
                data_loaders[dataset_name] = \
                    torch.utils.data.DataLoader(image_loaders['train'], batch_size=args.batch_size,
                                                shuffle=args.shuffle, num_workers=args.workers)
            elif dataset_name == 'val':
                data_loaders[dataset_name] = \
                    torch.utils.data.DataLoader(image_loaders['val'], batch_size=args.batch_size,
                                                shuffle=args.shuffle, num_workers=args.workers)
            elif dataset_name == 'test':
                data_loaders[dataset_name] = \
                    torch.utils.data.DataLoader(image_loaders['test'], batch_size=args.batch_size,
                                                shuffle=args.shuffle, num_workers=args.workers)
            elif dataset_name == 'viz':
                data_loaders[dataset_name] = \
                    torch.utils.data.DataLoader(image_loaders['viz'], batch_size=args.batch_size,
                                                shuffle=args.shuffle, num_workers=args.workers)
    return data_loaders


def save_gradient_image(gradient_data, greyscale=False):
    """
    save_gradient_image:
    :param gradient_data: A torch.cuda.FloatTensor or (if CUDA disabled a torch.FloatTensor) representing the gradient
        of the weights with respect to the input image.
    :param greyscale: A boolean flag indicating whether to save a grayscale gradient or a colored gradient image.
    :return:
    """
    output_path = os.path.join(str.replace(args.data, 'data', 'results'))
    # Duplicate the directory structure of args.data to store output images:
    for item in os.listdir(os.path.join(args.data, 'viz')):
        s = os.path.join(args.data, item)
        d = os.path.join(output_path, item)
        if not os.path.isdir(d):
            os.mkdir(d)
        if args.viz == 'vanilla_backprop':
            if greyscale:
                gradient_data = convert_to_grayscale(gradient_data)
                output_file_path = os.path.join(d, 'vanilla_bp_gray.jpg')
            else:
                output_file_path = os.path.join(d, 'vanilla_bp_color.jpg')
            # normalize:
            gradient_data = gradient_data - gradient_data.min()
            gradient_data /= gradient_data.max()
            gradient_data = np.uint8(gradient_data * 255).transpose(1, 2, 0)
            gradient_data = gradient_data[..., ::-1]

        # write image to hard drive
        cv2.imwrite(output_file_path, gradient_data)
        # shutil.copytree(s, d, symlinks=False, ignore=None)
        # else:
        #     os.mkdir(d)

def main():
    dataset_is_present = {
        'train': os.path.isdir(os.path.join(args.data, 'train')),
        'val': os.path.isdir(os.path.join(args.data, 'val')),
        'test': os.path.isdir(os.path.join(args.data, 'test')),
        'viz': os.path.isdir(os.path.join(args.data, 'viz'))
    }
    # Load pipelines for data transformation:
    data_transforms = get_data_transforms(dataset_is_present)
    # Load pipelines for image dataset loaders:
    image_loaders = get_image_loaders(data_transforms)
    dataset_sizes = {x: len(image_loaders[x]) for x in list(image_loaders.keys())}

    # Class names:
    class_names = None
    if dataset_is_present['train']:
        class_names = [int(imgnet_class_id) for imgnet_class_id in image_loaders['train'].classes]
        print('class_names: ', class_names)
    elif dataset_is_present['viz']:
        class_names = [int(imgnet_class_id) for imgnet_class_id in image_loaders['viz'].classes]
        print('class_names: ', class_names)
    else:
        print('Critical Error: Couldn\'t instantiate an image_loader for either the training or visualization dataset.'
              'Impossible to infer class names/labels with no image_loaders. Program terminating.')
        exit(-1)
    # Data Loaders:
    data_loaders = get_data_loaders(image_loaders)

    model = None
    # Create the model:
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        pretrained_model = models.__dict__[args.arch](pretrained=True)
        if use_gpu:
            pretrained_model = pretrained_model.cuda()
    else:
        print("=> creating model '{}'".format(args.arch))
        pretrained_model = models.__dict__[args.arch]()
        print("=> training '{}' from scratch".format(args.arch))
        print('=> CUDA is enabled?: %s\n=> Will use GPU to train?: %s' % (use_gpu, use_gpu))
        # Train the network:
        # pretrained_model = train(net=model, train_loader=data_loaders['train'], criterion=criterion, optimizer=optimizer)
        return NotImplementedError

    if args.viz == 'vanilla_backprop':
        model = VanillaBackprop(model=pretrained_model)
    elif args.viz == 'guided_backprop':
        # GBP = GuidedBackprop(model=pretrained_model)
        return NotImplementedError

    '''
    Visualize:
    '''
    # Retrieve all visualization data:
    for i, data in enumerate(data_loaders['viz']):
        images, labels = data
        # Wrap input in autograd Variable:
        if use_gpu:
            images, labels = Variable(images.cuda(), requires_grad=True), Variable(labels.cuda())
        else:
            images, labels = Variable(images, requires_grad=True), Variable(labels)
        for j, (image, label) in enumerate(zip(images, labels)):
            gradient = model.compute_gradients_for_single_image(image, class_names[j])
            # Save the image:
            save_gradient_image(gradient.data, greyscale=False)
            save_gradient_image(gradient.data, greyscale=True)

if __name__ == '__main__':
    main()
