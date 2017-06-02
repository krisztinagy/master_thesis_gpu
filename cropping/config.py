image = dict(
	# height = 64,
	# width = 64,
	# colorspace = 'grayscale',
	channels = 1,
	# format = 'JPEG',
	# noise_scale = 1,
)
dataset = dict(
    label0_folder = 'label0',
    label1_folder = 'label1',
	# size = 100000,
	# train_ratio = 0.8,
	# test_ratio = 0.2,
	num_categories = 2,

 )
dir = dict(
    data = '/pqry/data',
    crops = '/pqry/data/117_amneville_cropped',
    full_images = '/pqry/data/117',
    
	# output = '/home/nagy729krisztina/cropping/locarno_cropped/output',
    # tfrecords = '/home/nagy729krisztina/cropping/locarno_cropped/output/' + '64/' + str(dataset['size']),
)
files = dict(
    json = 'mentis-amneville-one.json',
    csv = 'occupancy117.csv'
)
processing = dict(
	training_shards = 100,
	testing_shards = 100,
    threads = 10,
)

threads = dict(
	preprocess = 1,
)

# tfRecords = dict(
	# prefix = 'try',
	# training_file = 'train-00000-of-00001',
	# testing_file = 'validation-00000-of-00001',
# )

# hyperparameters = dict(
	# learning_rate = 0.01,
	# batch_size = 10,
	# num_epochs = 1,
	# num_epochs_eval = 1,
# )

# # specify the neural network and the loss function for training
# # available models can be found in models/
# # available loss functions can be found in loss_functions/
# model = dict(
	# model_import = 'vgg19_trainable',
	# loss_function_import = 'cross_entropy',
# )