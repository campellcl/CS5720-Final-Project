"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms, utils
from src.misc_functions import get_params, convert_to_grayscale, save_gradient_images

'''
Command Line Argument Parsing
'''
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Visualizing CNN Gradients via Colored Standard Backpropagation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture ' + ' | '.join(model_names) + ' (default: alexnet)')
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
dataset_sizes = None


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


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


class MyVanillaBackprop():
    """
    MyVanillaBackprop: Performs standard backpropagation using the pre-trained model provided during instantiation.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        """

        :return:
        """

        def module_gradient_hook(module, grad_in, grad_out):
            """
            module_gradient_hook: This function hook is executed during backpropagation and captures the gradients of the
                hooked module with respect to the input images. The captured gradients are stored in this wrapper classes'
                self.gradients. This hook will be executed when backward is called on the source model.
            :param module: The layer in the network for which the hook has been attached to; the callee of this method.
            :param grad_in: A 3-tuple containing the gradients with respect to the inputs propagated in the backward
                direction.
            :param grad_out: A 1-tuple containing the gradients exiting the hooked module in the backward direction.
            :return None: Upon execution, the self.gradients field of the wrapper class will be propagated with grad_in[0].
            """
            print('Now executing module_hook on module: %s' % module)
            print('Inside ' + self.model.__class__.__name__ + ' backward')
            print('Inside class: ' + self.model.__class__.__name__)
            print('')
            print('grad_input type: %s of size %d' % (type(grad_in), len(grad_in)))
            if grad_in[0] is not None:
                print('\tgrad_input[0] type: ', type(grad_in[0]))
                print('\tgrad_input[0] size: ', grad_in[0].size())
            else:
                print('\tgrad_input[0] type: None')
                print('\tgrad_input[0] size: None')
            if grad_in[1] is not None:
                print('\tgrad_input[1] type: ', type(grad_in[1]))
                print('\tgrad_input[1] size: ', grad_in[1].size())
            else:
                print('\tgrad_input[1] type: None')
                print('\tgrad_input[1] size: None')
            if grad_in[2] is not None:
                print('\tgrad_input[2] type: ', type(grad_in[2]))
                print('\tgrad_input[2] size: ', grad_in[2].size())
            else:
                print('\tgrad_input[2] type: None')
                print('\tgrad_input[2] size: None')
            print('')
            print('grad_output type: %s of size %d' % (type(grad_out), len(grad_out)))
            if grad_out[0] is not None:
                print('\tgrad_output[0] type: ', type(grad_out[0]))
                print('\tgrad_output[0]: ', grad_out[0].size())
            else:
                print('\tgrad_output[0] type: None')
                print('\tgrad_output[0] size: None')
            self.gradients = grad_in[0]

        # Hook the first layer to get the gradient:
        # See: http://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks
        # See: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/23?u=campellcl
        '''
        "If you care about grad or outputs of some module, use module hooks, if you care about the grad w.r.t. some 
            Variable attach a hook to a Variable (there already is a register_hook function)".
        '''
        # Insert a module hook to capture the gradient of the first layer during backpropagation:
        if self.model.features is not None:
            first_layer = self.model._modules['features'][0]
            first_layer.register_backward_hook(module_gradient_hook)
        else:
            network_layers_odict_keys = list(self.model._modules.keys())
            first_layer = self.model._modules.get(network_layers_odict_keys[0])
            first_layer.register_backward_hook(module_gradient_hook)

    def compute_gradients_for_single_image(self, input_image, target_class_label):
        """
        compute_gradients_for_single_image: Computes the gradients for a single input image via one_hot_encoding and
            backpropagation.
        :param input_image: The image read by the dataloader for which the gradients are to be computed.
        :param target_class_label: The target class label of the supplied image.
        :return gradient_array: A numpy array populated by the gradient of the input image.
        """
        gradient_array = None
        # Have to add an extra preceding dimension for the tensor of a single image:
        input_image = input_image.resize(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
        # Compute forward pass through the model for this single image:
        output = self.model(input_image)
        print('model params: %s' % self.model.parameters)
        print('output\'s grad_fn: %s' % output.grad_fn)
        final_layer_weights, preds = torch.max(output.data, 1)
        # Zero the parameter gradients:
        # TODO: Not clear if the gradients of the optimizer or the model itself should be zeroed.
        # See: https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887
        self.model.zero_grad()
        # Create a target vector for backpropagation:
        # I believe we are setting the value to one because:
        #   https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments/47026836#47026836
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class_label] = 1
        # Compute the backward pass using the supplied gradient parameter
        output.cpu().backward(gradient=one_hot_output)
        # Convert PyTorch autograd.Variable into numpy array; [0] gets rid of the first channel (1, 3, 224, 224):
        gradient_array = self.gradients.data.numpy()[0]
        return gradient_array


def main(bulak_image):
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
    if dataset_is_present['train']:
        class_names = image_loaders['train'].classes
    elif dataset_is_present['viz']:
        class_names = image_loaders['viz'].classes
    print('class_names: ', class_names)
    # Data Loaders:
    data_loaders = get_data_loaders(image_loaders)
    # Define loss function (criterion) to evaluate how far off the network is in its predictions:
    criterion = nn.CrossEntropyLoss

    # Define the optimizer:
    # TODO: Does a pretrained module have a optimizer already associated? If so, it's being overwritten here:
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD

    # Create the model:
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        pretrained_model = models.__dict__[args.arch](pretrained=True)
        if use_gpu:
            pretrained_model = pretrained_model.cuda()
        MVBP = MyVanillaBackprop(model=pretrained_model)
    else:
        print("=> creating model '{}'".format(args.arch))
        pretrained_model = models.__dict__[args.arch]()
        print("=> training '{}' from scratch".format(args.arch))
        print('=> CUDA is enabled?: %s\n=> Will use GPU to train?: %s' % (use_gpu, use_gpu))
        # Train the network:
        # pretrained_model = train(net=model, train_loader=data_loaders['train'], criterion=criterion, optimizer=optimizer)
        MVBP = MyVanillaBackprop(model=pretrained_model)
        return NotImplementedError

    '''
    Visualize:
    '''
    # Apply a wrapper to the model to register function hooks for visualization:
    # VBP = VanillaBackprop(model)
    # Retrieve all visualization data:
    for i, data in enumerate(data_loaders['viz']):
        images, labels = data
        # Wrap input in autograd Variable:
        if use_gpu:
            images, labels = Variable(images.cuda(), requires_grad=True), Variable(labels.cuda())
        else:
            images, labels = Variable(images, requires_grad=True), Variable(labels)
        # print(bulak_image.cuda().equal(images))
        # outputs = model(images)
        # final_layer_weights, preds = torch.max(outputs.data, 1)
        gradient_array = MVBP.compute_gradients_for_single_image(images[0], 243)


if __name__ == '__main__':
    # Get params:
    target_example = 1  # Dog and cat
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example)
    main(bulak_image=prep_img)
    # print('pretrained_model._modules.keys():\n', pretrained_model._modules.keys())
    # print('')
    # print('pretrained_model._modules[\'features\']._modules:\n', pretrained_model._modules['features'])
    # print('')
    # print('pretrained_model._modules[\'classifier\']._modules:\n', pretrained_model._modules['classifier'])
    # print('')
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')
