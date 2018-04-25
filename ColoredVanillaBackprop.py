"""
ColoredVanillaBackprop.py

"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

'''
Command Line Argument Parsing:
'''

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Visualizing CNN Gradients via Colored Standard Backpropagation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# trans_learn (int, optional): Transfer learning enabled? Set to True to have only the last n fully connected layers
#   frozen. (default: False)
# parser.add_argument('--trans_learn', '-tn', metavar='TRANS_LEARN', type=str, help='number of fully connected layers to re-train')
#
# Args passed in to command line:
args = parser.parse_args()
# if args.trans_learn:
#     if args.trans_learn == 'ft':
#         print('Transfer learning enabled, fine tuning')

'''
Global Variables:
'''
# Flag indicating if GPU should be used for training:
use_gpu = torch.cuda.is_available()
# Flag indicating if test set is present (if not present, usually only a validation dataset was provided):
has_test_set = False
best_prec1 = None


def get_data_transforms(input_load_size=(256, 256), receptive_field_size=(224, 224)):
    """
    get_data_transforms: Gets the data transformation pipeline for each of the three input datasets.
    :param input_load_size: The dimensions (x, y) for which the input image is to be re-sized to.
    :param receptive_field_size: The dimensions (x, y) for which the input image is to cropped to (either a center or
        randomized crop depending on the defined pipeline).
    :return data_transforms: A dictionary housing sequential function calls that perform transformations to images in
        each of the datasets: train, val, and test.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(receptive_field_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_load_size),
            transforms.CenterCrop(receptive_field_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_load_size),
            transforms.CenterCrop(receptive_field_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def get_image_loaders(data_transforms):
    """
    get_image_loaders: Returns image loaders for all three datasets if present. This is an intermediary step to creating
        a torch.utils.data.DataLoader which handles asynchronous mini-batch loading.
    :param data_transforms: A dictionary housing sequential function calls that perform transformations to images in
        each of the datasets: train, val, and test.
    :return image_loaders: A dictionary containing references to the image loaders for each dataset.
    """
    # Training set image loader:
    trainset_imgloader = datasets.ImageFolder(os.path.join(args.data, 'train'), data_transforms['train'])
    # Validation set image loader:
    valset_imgloader = datasets.ImageFolder(os.path.join(args.data, 'val'), data_transforms['val'])
    image_loaders = {
        'train': trainset_imgloader,
        'val': valset_imgloader,
    }
    # Testing set image loader:
    if has_test_set:
        testset_imgloader = datasets.ImageFolder(os.path.join(args.data, 'test'), data_transforms['test'])
        image_loaders['test'] = testset_imgloader
    return image_loaders


def get_data_loaders(data_transforms, image_loaders):
    """
    get_data_loaders: Returns a dict of data loader references, one for each dataset: train, val, test. The instances
        of torch.utils.data.DataLoader handles asynchronous loading of mini-batches.
    :param data_transforms: A dictionary housing sequential function calls that perform transformations to images in
        each of the datasets: train, val, and test.
    :param image_loaders: A dictionary containing references to the image loaders for each dataset.
    :return data_loaders: A dictionary of data loader instance references, one for each dataset: train, val, and test.
    """
    data_loaders = {}
    data_loaders['train'] = torch.utils.data.DataLoader(image_loaders['train'], batch_size=args.batch_size,
                                                        shuffle=args.shuffle, num_workers=args.workers)
    data_loaders['val'] = torch.utils.data.DataLoader(image_loaders['val'], batch_size=args.batch_size,
                                                      shuffle=args.shuffle, num_workers=args.workers)
    if has_test_set:
        data_loaders['test'] = torch.utils.data.DataLoader(image_loaders['test'], batch_size=args.batch_size,
                                                           shuffle=args.shuffle, num_workers=args.workers)
    return data_loaders


def train(net, train_loader):
    return NotImplementedError


def main():
    # Does the data directory contain a test set or just a validation set?
    has_test_set = os.path.isdir(os.path.join(args.data, 'test'))
    data_transforms = get_data_transforms()
    image_loaders = get_image_loaders(data_transforms)
    dataset_sizes = {x: len(image_loaders[x]) for x in list(image_loaders.keys())}
    class_names = class_names = image_loaders['train'].classes
    # Data Loaders:
    data_loaders = get_data_loaders(data_transforms, image_loaders)
    # Create the model:
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True).cuda()
        if use_gpu:
            model = model.cuda()
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        print("=> training '{}' from scratch".format(args.arch))

    # Define loss function (criterion) to evaluate how far off the network is in its predictions:
    if use_gpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # Define the optimizer:
    optimizer = torch.optim.SGD(model, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume from checkpoint if flag is present:
    if args.resume:
        # Is the path to the checkpoint valid?
        if os.path.isfile(os.path.join(args.resume)):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec_1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('=> CUDA is enabled?: %s\n=> Will use GPU to train?: %s' % (use_gpu, use_gpu))
    # Train the network:
    train(net=model, train_loader=data_loaders['train'])



if __name__ == '__main__':
    main()

