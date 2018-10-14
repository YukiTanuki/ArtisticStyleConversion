# input images path
#print('base image path : ')
#base_image_path = input()
#print('style image path : ')
#style_image_path = input()
#print('result prefix : ')
#result_prefix = input()
#print('Transform from ' + base_image_path + ' into style ' + style_image_path + ' and save as ' + result_prefix)

base_image_path = 'base/base1.jpg'
style_image_path = 'style/style1.jpg'
result_prefix = 'res/res'

from scipy.misc import imread, imresize
import numpy as np

img_width = 400
img_height = 400
assert img_height == img_width, 'Due to the use of the Gram matrix, width and height must match.'

def preprocess_image(image_path):
	# read images and resize
    img = imresize(imread(image_path, mode = 'RGB'), (img_width, img_height, 3))
    # RGB->BGR
    img = img[:, :, ::-1].astype(np.float64)
    # make the averages zero to use Caffe-VGG
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
	# add axis to use VGG
    img = np.expand_dims(img, axis = 0)
    return img

def deprocess_image(x):
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	# BGR->RGB
	x = x[:, :, ::-1]
    # clip
	x = np.clip(x, 0, 255).astype(np.uint8)
	return x



from keras import backend as K

base_image = K.variable(preprocess_image(base_image_path)) # content
style_image = K.variable(preprocess_image(style_image_path)) # style
# placeholder(1x(width)x(height)x3)
combination_image = K.placeholder((1, img_width, img_height, 3))
# concatnate inputs
input_tensor = K.concatenate([base_image,
							  style_image,
							  combination_image], axis = 0)



# VGG
#from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
model = VGG16(include_top = False, weights = 'imagenet', input_tensor = input_tensor)
#print(model.summary())
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
#print(outputs_dict)

def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features - 1, K.transpose(features - 1))
    return gram

def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
    b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

total_variation_weight = 1e-3
style_weight = 1.
content_weight = 0.025

loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(base_image_features, combination_features)

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
for layer_name in feature_layers:
	layer_features = outputs_dict[layer_name]
	style_reference_features = layer_features[1, :, :, :]
	combination_features = layer_features[2, :, :, :]
	sl = style_loss(style_reference_features, combination_features)
	loss += (style_weight / len(feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)
f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, img_width, img_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype(np.float64)
    else:
        grad_values = np.array(outs[1:]).flatten().astype(np.float64)
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
import time

x = preprocess_image(base_image_path) # content
#x = preprocess_image(style_image_path) # style
#x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) # white noise
x[0, :, :, 0] -= 103.939
x[0, :, :, 1] -= 116.779
x[0, :, :, 2] -= 123.68

# L-BFGS
for i in range(10):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime = evaluator.grads, maxfun = 20)
    print('Current loss value:', min_val)
    
    img = deprocess_image(x.copy().reshape((img_width, img_height, 3)))
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))