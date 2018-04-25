"""
ColoredVanillaBackprop.py

"""
import argparse
import os
import time
import copy
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt

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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
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
dataset_sizes = None

class AverageMeter(object):
    """
    AverageMeter: Computes and stores the average and current value during each epoch.
    source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VanillaBackprop(nn.Module):
    """
    VanillaBackprop: Performs standard backpropagation using the pre-trained model provided during instantiation.
    """
    def __init__(self, model):
        super(VanillaBackprop, self).__init__()
        print('[VanillaBackprop <wrapper>] Target architecture %s layers/modules (hook-able): %s' % (args.arch, model._modules.keys()))
        # This is a reference to the pre-trained source model (i.e. Resnet18):
        self.model = model
        # This is where we will store the gradients of the network for visualization purposes:
        self.gradients = None
        # Freeze the network gradients in every layer:
        for param in model.parameters():
            param.requires_grad = False
        # Put the model in evaluation mode:
        self.model.eval()
        # Hook the first layer to get the gradient:
        # See: http://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks
        # See: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/23?u=campellcl
        '''
        "If you care about grad or outputs of some module, use module hooks, if you care about the grad w.r.t. some 
            Variable attach a hook to a Variable (there already is a register_hook function)".
        '''
        # Get the first layer of the network:
        network_layers_odict_keys = list(self.model._modules.keys())
        first_layer = self.model._modules.get(network_layers_odict_keys[0])
        gradient_hook = first_layer.register_forward_hook(
            lambda layer, i, o:
                print(
                    'layer/module:', type(layer),
                    '\ni:', type(i),
                        '\n   len:', len(i),
                        '\n   type:', type(i[0]),
                        '\n   data size:', i[0].data.size(),
                        '\n   data type:', i[0].data.type(),
                    '\no:', type(o),
                        '\n   data size:', o.data.size(),
                        '\n   data type:', o.data.type(),
                )
        )

    def forward(self, *input):
        # Delegate to the super function of the provided pre-trained model parameter during instantiation:
        return self.model.forward(*input)
        # return super(nn.Module, self.model).forward(*input)
        # return super(type(nn.Module), self.model).forward(*input)
        # return super(VanillaBackprop, self).forward(*input)


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
                transforms.Resize(input_load_size),
                transforms.CenterCrop(receptive_field_size),
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
                                                shuffle = args.shuffle, num_workers=args.workers)
    return data_loaders


def save_checkpoint(state, is_best, filename='../PTCheckpoints/checkpoint.pth.tar'):
    """
    save_checkpoint: Saves the model's weights when called. If the is_best flag is set checkpoint will be saved as:
        model_best.pth.tar. If the flag is not set, checkpoint will be saved as 'checkpoint.pth.tar'.
    :param state: The information comprising the checkpoint.
    :param is_best: A boolean flag indicating if this is the best performing weights on the validation dataset.
    :param filename: The location and name of the checkpoint file to be written to disk.
    :return None: Upon completion, a checkpoint comprised of 'state' is saved to 'filename'.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(net, train_loader, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.SGD):
    """
    train: Trains a PyTorch neural network (nn.Module subclass) via the provided optimizer (torch.optim), assessed using
        the provided criterion, over num_epoch iterations.
    :param net: A nn.Module instance representing the neural network.
    :param train_loader: A nn.DataLoader instance which performs asynchronous loading from the training dataset.
    :param criterion: A loss function computing how far away the network's output is from the target.
    :param optimizer: An optimization function (such as Stochastic Gradient Descent) which performs updates to the
        network's weights based on the output of the criterion function.
    :return net: The provided network is returned after training with the best model weights found during validation.
    """
    top1 = AverageMeter()
    top5 = AverageMeter()
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    losses = []
    accuracies = []
    # Declare the loss (cost) function:
    criterion = criterion()
    # Declare the optimization function:
    optimizer = optimizer(net.parameters(), lr=args.lr, momentum=args.momentum)
    # Iterate over the dataset num_epoch times:
    for epoch in range(args.epochs):
        # Each epoch has a training and validation phase:
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            print('Epoch {}/{}'.format(epoch, args.epochs - 1))
            print('-' * 10)
            for i, data in enumerate(train_loader):
                # get the inputs:
                images, labels = data
                # wrap inputs in an autograd.Variable:
                if use_gpu:
                    images, labels = Variable(images.cuda()), Variable(labels.cuda())
                else:
                    images, labels = Variable(images), Variable(labels)
                # Zero the parameter gradients:
                optimizer.zero_grad()
                # Compute forward pass:
                outputs = net(images)
                # Get weights and predictions:
                weights, preds = torch.max(outputs.data, 1)
                # Compute the loss:
                loss = criterion(outputs, labels)
                # Backpropagate:
                loss.backward()
                # Update the network's weights with the update/learning rule:
                optimizer.step()
                # Print statistics every print_freq epochs:
                if i % args.print_freq == 0:
                    print('<epoch, batch>:\t\t[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.batch_size))
                    running_loss += loss.data[0] * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            losses.append(epoch_loss)
            epoch_acc = running_corrects / dataset_sizes[phase]
            accuracies.append(epoch_acc)
            print('[{}]:\t Epoch Loss: {:.4f} Epoch Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # Deep copy the model's weights if this epoch was the best peforming on the val set:
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                print('Checkpoint: Epoch %d has the best validation accuracy. The model weights have been saved.' % epoch)
                # Create checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch' : args.arch,
                    'state_dict': copy.deepcopy(net.state_dict()),
                    'best_prec_1': best_acc,
                    'optimizer': optimizer.state_dict()
                }, is_best=True, filename='../data/PTCheckpoints/model_best.pth.tar')
            print('Accuracy (Top-1 Error or Precision at 1) of the network on %d %s images: %.2f%%'
                  % (dataset_sizes[phase], phase, epoch_acc * 100))
            print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights:
    net.load_state_dict(best_model_wts)
    return net


def imshow_tensor(images, title=None):
    """
    imshow_tensor: Displays a tensor as a grid of images. The MatPlotLib imshow function for PyTorch Tensor Objects.
    :source url: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    :param images: Tensor images to be visualized.
    :param title: A list of titles each one corresponding to an image in images.
    :return:
    """
    # See: http://pytorch.org/docs/master/torchvision/utils.html#torchvision.utils.make_grid
    torchvision_grid = utils.make_grid(images)
    # TODO: not sure what the point of this transposition is:
    input = torchvision_grid.numpy().transpose((1, 2, 0))
    # Normalize the input Tensor by each input channel (R, G, B)'s mean and standard deviation:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    # Restrict to [0, 1] interval:
    input = np.clip(input, a_min=0, a_max=1)
    fig = plt.figure()
    has_title = title is not None
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.show()


def main():
    dataset_is_present = {}
    has_train_set = os.path.isdir(os.path.join(args.data, 'train'))
    has_val_set = os.path.isdir(os.path.join(args.data, 'val'))
    # Does the data directory contain a test set or just a validation set?
    has_test_set = os.path.isdir(os.path.join(args.data, 'test'))
    has_viz_set = os.path.isdir(os.path.join(args.data, 'viz'))
    dataset_is_present['train'] = has_train_set
    dataset_is_present['val'] = has_val_set
    dataset_is_present['test'] = has_test_set
    dataset_is_present['viz'] = has_viz_set
    del has_train_set, has_val_set, has_test_set, has_viz_set
    # Load pipelines for data transformation:
    data_transforms = get_data_transforms(dataset_is_present)
    # Load pipelines for image dataset loaders:
    image_loaders = get_image_loaders(data_transforms)
    dataset_sizes = {x: len(image_loaders[x]) for x in list(image_loaders.keys())}
    if dataset_is_present['train']:
        class_names = image_loaders['train'].classes
    elif dataset_is_present['viz']:
        class_names = image_loaders['viz'].classes

    # Data Loaders:
    data_loaders = get_data_loaders(image_loaders)

    # Define loss function (criterion) to evaluate how far off the network is in its predictions:
    if use_gpu:
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.CrossEntropyLoss().cuda
    else:
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss

    # Define the optimizer:
    # TODO: Does a pretrained module have a optimizer already associated? If so, it's being overwritten here:
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD

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
        print('=> CUDA is enabled?: %s\n=> Will use GPU to train?: %s' % (use_gpu, use_gpu))

        # Train the network:
        model = train(net=model, train_loader=data_loaders['train'], criterion=criterion, optimizer=optimizer)

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

    '''
    Visualize:
    '''
    # Apply a wrapper to the model to register function hooks for visualization:
    VBP = VanillaBackprop(model)
    # Retrieve all visualization data:
    for i, data in enumerate(data_loaders['viz']):
        images, labels = data
        # Wrap input in autograd Variable:
        if use_gpu:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        # Zero the parameter gradients:
        # optimizer.zero_grad()
        # Compute forward pass/generate gradients:
        outputs = VBP(images)
        # Get weights and predictions:
        weights, preds = torch.max(outputs.data, 1)


if __name__ == '__main__':
    main()

