# This script works only on python 3.4, with keras 1.2.2, Theano 1.0.0 and weight file
# vgg16_weights_th_dim_ordering_th_kernels.h5. 
# Usage of this script:
# 1. Install Python 3.4 (Keras 1.2.2 won't work well with Theano with py3.5+. Weird problems occur)
# 2. For Windows, download numpy+mkl, scipy and libpython for py34 
#    (make sure version is correct) from https://www.lfd.uci.edu/~gohlke/pythonlibs/ and install them with pip
# 3. Download keras 1.2.2 from https://pypi.python.org/pypi/Keras/1.2.2 and theano from 
#    https://github.com/Theano/Theano then install them by something like "python setup.py install"
# 4. Download vgg16_weights_th_dim_ordering_th_kernels.h5
# 5. Download this script and copy a random picture into the same directory as the weight file. 
# 6. Run the script and enjoy!

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications import imagenet_utils
from keras import backend as K
K.set_model_dim_ordering('th')
# importing deeplift functions
import deeplift
from deeplift.blobs import NonlinearMxtsMode
from deeplift.conversion import keras_conversion as kc
from deeplift.util import compile_func
from deeplift.util import get_integrated_gradients_function
# importing other functions
import cv2
import numpy as np
from copy import deepcopy
from collections import OrderedDict

# 3 util functions
def calc_threshold(score_matrix, method='32'):
	score_array = score_matrix.flatten()
	if (method == 'mean'):
		return np.mean(score_array)
	elif (method == '95_percentage'):
		score_array = np.sort(score_array)
		l = len(score_array)
		return score_array[int(l * 0.95)]
	elif (method == '90_percentage'):
		score_array = np.sort(score_array)
		l = len(score_array)
		return score_array[int(l * 0.9)]
	elif (method == '80_percentage'):
		l = len(score_array)
		score_array = np.sort(score_array)
		return score_array[int(l * 0.8)]
	else:
		return float(method)#raise RuntimeError("No such threshold calculation method: " + method)
	
def get_box_from_mask(flat_mask, original_size):
	W_orig, H_orig = original_size
	#print(original_size)
	
	col_max = np.max(flat_mask, axis = 0)
	row_max = np.max(flat_mask, axis = 1)
	
	col_idxs = np.where(col_max>0)
	xmin = col_idxs[0][0]
	xmax = col_idxs[0][-1]
	
	row_idxs = np.where(row_max>0)
	ymin = row_idxs[0][0]
	ymax = row_idxs[0][-1]
	
	xmin = int(xmin*W_orig/224)
	xmax = int(xmax*W_orig/224)
	ymin = int(ymin*H_orig/224)
	ymax = int(ymax*H_orig/224) 
	
	bbox = (xmin, xmax, ymin, ymax)
	return bbox
	
def cv_read_image_pair(image_path, reference_path, reference_default):
	# pre-treat image data
	im = cv2.imread(image_path)
	orig_w = im.shape[1]
	orig_h = im.shape[0]
	#print((orig_w, orig_h))
	im = cv2.resize(im, (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	# get reference
	if (reference_path == None):
		if (reference_default == 'zero'):
			reference = np.zeros_like(im)
		elif (reference_default == 'blur'):
			kernel = np.array([[0.003765,0.015019,0.023792,0.015019,0.003765], 
			[0.015019,0.059912,0.094907,0.059912,0.015019],
			[0.023792,0.094907,0.150342,0.094907,0.023792],
			[0.015019,0.059912,0.094907,0.059912,0.015019],
			[0.003765,0.015019,0.023792,0.015019,0.003765]])
			reference = cv2.filter2D(im,-1,kernel)
			reference = cv2.filter2D(reference,-1,kernel)
			reference = cv2.filter2D(reference,-1,kernel)
		else:
			print("Warning: Invalid reference generator. Reference is set to None.")
			reference = None
	else:
		reference = cv2.resize(cv2.imread(reference_path), (224, 224)).astype(np.float32)
	# after_treat
	im = im.transpose((2,0,1))
	reference = reference.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	reference = np.expand_dims(reference, axis=0)
	return (im, reference, orig_w, orig_h)
	
# VGG16 and VGG16-DeepLIFT
def VGG_16(weights_path=None):
	# standard VGG16 model. Architected as configuration D of
	# Very Deep Convolutional Networks for Large-Scale Image Recognition.
	# K. Simonyan, A. Zisserman. arXiv:1409.1556
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(3, 224, 224)))
	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	
	model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	
	model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(1000, activation='softmax'))
	
	if weights_path:
		model.load_weights(weights_path)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	return model
	
def VGG_16_Predict(data_path, reference_path=None, reference_default='blur', weights_path=None):
	# read data
	data, reference, w, h = cv_read_image_pair(data_path, reference_path, reference_default)
	# load Keras model and perform prediction
	keras_model = VGG_16(weights_path=weights_path)
	print("Model loaded successfully. Predicting values...")
	
	print("For original data:")
	pr_out = keras_model.predict(data)
	P = imagenet_utils.decode_predictions(pr_out)
	print("\nPredictions are:")
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
		
	print("For reference data:")
	pr_out = keras_model.predict(reference)
	P = imagenet_utils.decode_predictions(pr_out)
	print("\nPredictions are:")
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
	
def VGG_16_DeepLIFT_model(keras_model, analytic='revealcancel'):
	# select DeepLIFT model
	print("\nGenerating DeepLIFT model and functions...")
	if (analytic == 'rescale'):
		deeplift_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.Rescale)
		deeplift_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)
	elif (analytic == 'revealcancel'):
		deeplift_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)
		deeplift_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)
	elif (analytic == 'rescale_conv_revealcancel_fc'):
		deeplift_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
		deeplift_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)
	elif (analytic == 'gradient_times_input'):
		deeplift_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.Gradient)
		deeplift_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-2)
	elif (analytic == 'integrated_gradient_5'):
		deeplift_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.Gradient)
		gradient_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-2)
		deeplift_func = get_integrated_gradients_function(gradient_func, 5)
	elif (analytic == 'integrated_gradient_10'):
		deeplift_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.Gradient)
		gradient_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-2)
		deeplift_func = get_integrated_gradients_function(gradient_func, 10)
	elif (analytic == 'guided_backprop_times_input'):
		deeplift_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.GuidedBackprop)
		deeplift_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-2)
	else:
		raise RuntimeError("No such analytic method: " + method)
	print("Finished creating DeepLIFT functions.")	
	return((deeplift_model, deeplift_func))

def VGG_16_DeepLIFT_predict(data, reference, deeplift_tuple, keras_result):		
	# check validity of deepLIFT model
	deeplift_model, deeplift_func = deeplift_tuple
	deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()], deeplift_model.get_layers()[-1].get_activation_vars())
	converted_model_predictions = deeplift.util.run_function_in_batches(input_data_list=[data], func=deeplift_prediction_func, batch_size=32, progress_update=None)
	max_deeplift_diff = np.max(np.array(converted_model_predictions)-np.array(keras_result))
	if (max_deeplift_diff >= 10**-5):
		raise RuntimeError("DeepLIFT conversion differ from original model :" + str(max_deeplift_diff) + ". Please contact the author for more information.")
	
	# calculate scores and mask
	task_index = np.argmax(keras_result)
	scores = np.array(deeplift_func(task_idx=task_index, input_data_list=[data], input_references_list=[reference],
                    batch_size=32, progress_update=None))
	score_pic = scores[0,:,:,:] * (scores[0,:,:,:] > 0) / np.max(scores[0,:,:,:])
	print("Finished calculating score.")
	return score_pic.transpose((1,2,0))
	#filename = 'deeplift_' + analytic + "_" + reference_default + "_" + data_path
	#cv2.imwrite(filename, score_pic.transpose((1,2,0)))
	#print("Finished calculating mask and saved as: " + filename)
	

def VGG_16_Combined(data_paths, reference_paths=None, reference_default='blur', weights_path=None, mask_path='vgg16_mask.jpg', save_map=False):
	# load Keras and DeepLIFT model
	print("Loading DeepLIFT models...")
	keras_model = VGG_16(weights_path=weights_path)
	model_tuple_a = VGG_16_DeepLIFT_model(keras_model, analytic='rescale')
	model_tuple_b = VGG_16_DeepLIFT_model(keras_model, analytic='guided_backprop_times_input')
	print("Model loaded successfully. Predicting values...")
	
	for i in range(0,len(data_paths)):
		# read data and perform prediction
		data_path = data_paths[i]
		if (reference_paths):
			reference_path = reference_paths[i]
		else:
			reference_path = None
		data, reference, w, h = cv_read_image_pair(data_path, reference_path, reference_default)
		pr_out = keras_model.predict(data, batch_size=32)
		P = imagenet_utils.decode_predictions(pr_out)
		print("\nPredictions for " + data_path + " are:")
		for (i, (imagenetID, label, prob)) in enumerate(P[0]):
			print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
		
		# Gaussian smooth kernel
		kernel = np.array([[0.003765,0.015019,0.023792,0.015019,0.003765], 
			[0.015019,0.059912,0.094907,0.059912,0.015019],
			[0.023792,0.094907,0.150342,0.094907,0.023792],
			[0.015019,0.059912,0.094907,0.059912,0.015019],
			[0.003765,0.015019,0.023792,0.015019,0.003765]])
		# calculate contribution scores
		a = VGG_16_DeepLIFT_predict(data, reference, model_tuple_a, pr_out)
		a = cv2.filter2D(a, -1, kernel)
		a = cv2.filter2D(a, -1, kernel)
		b = VGG_16_DeepLIFT_predict(data, reference, model_tuple_b, pr_out)
		b_mask = cv2.imread(mask_path).astype(np.float32) / 255.0
		b = cv2.filter2D(b * b_mask, -1, kernel)
		b = cv2.filter2D(b, -1, kernel)
		b_threshold = calc_threshold(b, method='90_percentage')
		b = (b - b_threshold) * (b > b_threshold) * 255.0 / (255 - b_threshold)
		#a = cv2.imread("deeplift_combined_a_" + data_path).astype(np.float32) / 255.0
		#b = cv2.imread("deeplift_combined_b_" + data_path).astype(np.float32) / 255.0
		c = np.power(2*a*b/(a+b+1e-6), 0.5) * 255.0
		
		# save score file.
		if (save_map):
			filename = 'deeplift_combined_a_' + data_path
			cv2.imwrite(filename, a * 255.0)
			filename = 'deeplift_combined_b_' + data_path
			cv2.imwrite(filename, b * 255.0)
			filename = 'deeplift_combined_c_' + data_path
			cv2.imwrite(filename, c)
			print("Finished calculating mask and saved as: " + filename)
		c_threshold = calc_threshold(c, method='96')
		
		# calculate and implement the box
		xmin, xmax, ymin, ymax = get_box_from_mask(np.sum(c, axis=2) > c_threshold, (w, h))
		im_out = cv2.imread(data_path).astype(np.float32)
		print(im_out.shape)
		im_out[ymin:ymax,[xmin,xmax], 0] = 255
		im_out[ymin:ymax,[xmin,xmax], 1] = 255
		im_out[ymin:ymax,[xmin,xmax], 2] = 0
		im_out[[ymin,ymax],xmin:xmax, 0] = 255
		im_out[[ymin,ymax],xmin:xmax, 1] = 255
		im_out[[ymin,ymax],xmin:xmax, 2] = 0
		filename = 'output96_' + data_path
		cv2.imwrite(filename, im_out)
		print("Finished localization and result saved as: " + filename)
	
	
if __name__ == "__main__":
	VGG_16_Combined(['output_ILSVRC2012_test_00000022.jpeg', 'output_ILSVRC2012_test_00000036.jpeg', 'output_ILSVRC2012_test_00000039.jpeg'], weights_path='vgg16_weights_th_dim_ordering_th_kernels.h5', save_map=True)
	
