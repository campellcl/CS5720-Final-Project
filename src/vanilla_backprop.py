"""
vanilla_backprop.py
Performs vanilla backpropagation and returns the gradients of the input image for visualization.
Adapted from: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

__author__ = 'Chris Campell'
__created__ = '4/29/2018'

'''
Command Line Argument Parsing
'''
# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
# parser = argparse.ArgumentParser(description='Visualizing CNN Gradients via Colored Standard Backpropagation')
# parser.add_argument('data', metavar='DIR', help='path to dataset')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet', choices=model_names,
#                     help='model architecture ' + ' | '.join(model_names) + ' (default: alexnet)')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# # How many images will the data loader grab during one call to next(iter(data_loader)) or iter(data_loader).next()
# parser.add_argument('-b', '--batch_size', default=256, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# # Shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: False).
# parser.add_argument('--shuffle', '-s', default=False, type=bool, metavar='SHUFFLE',
#                     help='Reshuffle the data at every epoch? (default: False)')
# # num_workers (int, optional): how many subprocesses to use for data loading.
# #   0 means that the data will be loaded in the main process. (default: 4)
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# # Args passed in to command line:
# args = parser.parse_args()

'''
Global Variables:
'''
# Flag indicating if GPU should be used for training:
use_gpu = torch.cuda.is_available()


class VanillaBackprop():
    """
    VanillaBackprop: Performs standard backpropagation using the pre-trained model provided during instantiation.
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
        hook_layers: Attaches function hooks to modules/layers in the pretrained network which will fire during certain
            functions calls. For instance, backward hooks fire during backpropagation (i.e. output.backward()) and
            forward hooks fire during forward propagation (i.e. output = model(input)).
        :return None: Upon completion, hooks will be attached to certain layers to extract gradient during intermediary
            computations.
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
        # [0] gets rid of the first channel (1, 3, 224, 224):
        return self.gradients[0]
