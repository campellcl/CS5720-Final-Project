# CS5720-Final-Project
Final two visualizations for CS 5720 (Scientific Computing with Visualization). 

## Project Proposal
* [Proposal](https://docs.google.com/document/d/1_HCKTHz8wSD9y48NpfAA3F-t9V3hiOMYgpq0Dt18YxY/edit?usp=sharing)

## Visualization One:

## Visualization Two:
### Learning Resources:
* [Visualizing what ConvNets Learn](http://cs231n.github.io/understanding-cnn/)
* [Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
* [PyTorch CNN Visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* [Intel AI Academy: Visualising CNN Models Using PyTorch](https://software.intel.com/en-us/articles/visualising-cnn-models-using-pytorch)

### Implementation Resources:
* [PyTorch and Function Hooks](http://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks)
* [PyTorch and Function Hooks (2)]([https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/23)
* [PyTorch Tensors and the register_hook Method](http://pytorch.org/docs/master/autograd.html#torch.Tensor.register_hook)
* [Virginia PyTorch Lab](http://www.cs.virginia.edu/~vicente/vislang/notebooks/pytorch-lab.html)
* [How to Use torch.autograd.backward when Variables are Non-Scalar](https://discuss.pytorch.org/t/how-to-use-torch-autograd-backward-when-variables-are-non-scalar/4191)
* [Tensors that Track History](http://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html)
* [Clarification - Using backward() on non-scalars](https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059)
* [Exact meaning of grad_input and grad_output](https://discuss.pytorch.org/t/exact-meaning-of-grad-input-and-grad-output/14186/2)

#### Obtaining Gradients from Pre-Trained Models
##### Method One: Gradients for Only Parameters
1. `params = list(network.parameters())`
2. Now `params` is a list of `nn.Parameter` objects, which are a subclass of `Variable`. Each object in `params` will have a `.grad` attribute containing the gradients.
3. To access: `params[0].grad # gradient of the first module/layer's first weight`
4. Now we can get the gradients for the entire layer: `grad = [p.grad for p in list(network.parameters())]`
##### Method Two: Intermediate Gradients of Not Just Parameters
1. `register_backward_hook()`
* See [Virginia PyTorch Lab](http://www.cs.virginia.edu/~vicente/vislang/notebooks/pytorch-lab.html)

### Run Configurations:
* See attached RunConfigurations.txt file.


### Dataset:

