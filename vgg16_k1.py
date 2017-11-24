# This script works only on python 3.4, with keras 1.2.2, Theano 1.0.0 and weight file
# vgg16_weights_th_dim_ordering_th_kernels.h5
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


# two util functions
def calc_threshold(score_matrix, method='192'):
	score_array = score_matrix.flatten()
	if (method == 'mean'):
		return np.mean(score_array)
	elif (method == '80_percentage'):
		l = len(score_array)
		score_array = np.sort(score_array)
		return score_array[int(l * 0.8)]
	elif (method == '50_percentage'):
		l = len(score_array)
		score_array = np.sort(score_array)
		return score_array[int(l * 0.5)]
	else:
		return float(method)#raise RuntimeError("No such threshold calculation method: " + method)
	
def get_box_from_mask(flat_mask, original_size):
	W_orig, H_orig = original_size
	#print(original_size)
      
	col_max = np.max(flat_mask, axis = 0)
	row_max = np.max(flat_mask, axis = 1)
	
	col_idxs = np.where(col_max>0)
	ymin = int(col_idxs[0][0] * H_orig / 224.0)
	ymax = int(col_idxs[0][-1] * H_orig / 224.0)
	
	row_idxs = np.where(row_max>0)
	xmin = int(row_idxs[0][0] * W_orig / 224.0)
	xmax = int(row_idxs[0][-1] * W_orig / 224.0)
	
	bbox = (xmin, xmax, ymin, ymax)
	print((col_idxs[0][0], col_idxs[0][-1], row_idxs[0][0], row_idxs[0][-1]), bbox)
	return bbox
	
def cv_read_image_pair(image_path, reference_path, reference_default):
	# pre-treat image data
	im = cv2.imread(image_path)
	orig_w = im.shape[1]
	orig_h = im.shape[0]
	print((orig_w, orig_h))
	im = cv2.resize(im, (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	# get reference
	if (reference_path == None):
		if (reference_default == 'zero'):
			reference = np.zeros_like(im)
		elif (reference_default == 'blur'):
			kernel = np.ones((5,5),np.float32)/25
			reference = cv2.filter2D(im,-1,kernel)
			reference = cv2.filter2D(reference,-1,kernel)
			reference = cv2.filter2D(reference,-1,kernel)
			#sr = deepcopy(reference)
			#sr[:,:,0] += 103.939
			#sr[:,:,1] += 116.779
			#sr[:,:,2] += 123.68
			#cv2.imwrite('sr.png', sr)
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
	

def VGG_16_DeepLIFT(data, reference, keras_model, keras_result, analytic='revealcancel'):		
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
	
	# check validity of deepLIFT model
	deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()], deeplift_model.get_layers()[-1].get_activation_vars())
	converted_model_predictions = deeplift.util.run_function_in_batches(input_data_list=[data], func=deeplift_prediction_func, batch_size=32, progress_update=None)
	max_deeplift_diff = np.max(np.array(converted_model_predictions)-np.array(keras_result))
	if (max_deeplift_diff >= 10**-5):
		raise RuntimeError("DeepLIFT conversion differ from original model :" + str(max_deeplift_diff) + ". Please contact the author for more information.")
	
	# calculate scores and mask
	task_index = np.argmax(keras_result)
	scores = np.array(deeplift_func(task_idx=task_index, input_data_list=[data], input_references_list=[reference],
                    batch_size=32, progress_update=None))
	score_pic = scores[0,:,:,:] * (scores[0,:,:,:] > 0) / np.max(scores[0,:,:,:]) * 255.0
	print("Finished calculating score of " + analytic + " method.")
	return score_pic.transpose((1,2,0))
	#filename = 'deeplift_' + analytic + "_" + reference_default + "_" + data_path
	#cv2.imwrite(filename, score_pic.transpose((1,2,0)))
	#print("Finished calculating mask and saved as: " + filename)
	

def VGG_16_Combined(data_path, reference_path=None, reference_default='blur', weights_path=None, save_map=False):
	# read data
	data, reference, w, h = cv_read_image_pair(data_path, reference_path, reference_default)
	# load Keras model and perform prediction
	keras_model = VGG_16(weights_path=weights_path)
	print("Model loaded successfully. Predicting values...")
	pr_out = keras_model.predict(data, batch_size=32)
	P = imagenet_utils.decode_predictions(pr_out)
	print("\nPredictions are:")
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
	# calculate contribution scores
	a = VGG_16_DeepLIFT(data, reference, keras_model, pr_out, analytic='rescale')
	b = VGG_16_DeepLIFT(data, reference, keras_model, pr_out, analytic='guided_backprop_times_input')
	c = np.sqrt(a) * b
	# save score file.
	if (save_map):
		filename = 'deeplift_combined' + "_" + data_path
		cv2.imwrite(filename, c)
		print("Finished calculating mask and saved as: " + filename)
	# calculate the surrounding rectangle
	kernel = np.ones((5,5),np.float32)/25
	c = cv2.filter2D(c,-1,kernel)
	c_threshold = calc_threshold(c, method='192')
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
	filename = 'output_' + data_path
	cv2.imwrite(filename, im_out)
	print("Finished localization and result saved as: " + filename)
	
	
if __name__ == "__main__":
	VGG_16_Combined('snake.jpg', weights_path='vgg16_weights_th_dim_ordering_th_kernels.h5')
	
