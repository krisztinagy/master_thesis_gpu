image = dict(
	height = 32,
	width = 32,
	channels = 1,
)

dataset = dict(
	num_categories = 2,
	label0_folder = 'label0',
	label1_folder = 'label1',
    percentage_to_use = 1
)
testing = dict(
    crop_percentage = 0.8,
)

model = dict(
	vgg19_pretrained = '/pqry/pqry/data/vgg19.npy',
    model_import = 'M3',
    model_dir = 'M3b',
	loss_function_import = 'cross_entropy',
)

directory = dict(
	tfrecords_train = '/pqry/pqry/data/tfrecords_3slots/train',
	tfrecords_test = '/pqry/pqry/data/tfrecords_3slots/test',
	results = '/pqry/pqry/results_3slots',
    tensorboard = 'tensorboard_logs',
)

hyperparameters = dict(
	learning_rate = 0.01,
	batch_size = 50,
	num_epochs = 50,
	num_epochs_eval = 1,
    num_threads = 10,
)

log = dict(
    config_file = 'config.py',
)

