image = dict(
	height = 224,
	width = 224,
	channels = 1,
)

dataset = dict(
	num_categories = 2,
	label0_folder = 'label0',
	label1_folder = 'label1',
)

model = dict(
	vgg19_pretrained = '/pqry/data/vgg19.npy',
    #model_import = 'vgg_64_gray_wofc_compressed',
    model_import = 'M0',
	loss_function_import = 'cross_entropy',
)

directory = dict(
	tfrecords_train = '/pqry/data/tfrecords/train',
	tfrecords_test = '/pqry/data/tfrecords/test',
	results = '/pqry/results',
    tensorboard = 'tensorboard_logs',
)

hyperparameters = dict(
	learning_rate = 0.01,
	batch_size = 10,
	num_epochs = 10,
	num_epochs_eval = 1,
    num_threads = 10,
)

log = dict(
    config_file = 'config.py',
)

