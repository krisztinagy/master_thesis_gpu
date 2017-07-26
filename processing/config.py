image = dict(
	height = 16,
	width = 16,
	channels = 1,
)

dataset = dict(
	num_categories = 2,
    label0 = 'label0',
    label1 = 'label1',
    # label000_folder = 'label000',
    # label001_folder = 'label001',
    # label010_folder = 'label010',
    # label011_folder = 'label011',
    # label100_folder = 'label100',
    # label101_folder = 'label101',
    # label110_folder = 'label110',
    # label111_folder = 'label111',
    percentage_to_use = 0.2,
)
testing = dict(
    crop_percentage = 0.8,
)

model = dict(
	vgg19_pretrained = '/pqry/pqry/data/vgg19.npy',
    model_import = 'M3',
    model_dir = 'M3_1spot_cam170_16_16_02',
	loss_function_import = 'cross_entropy',
)

directory = dict(
	tfrecords_train = '/pqry/pqry/data/tfrecords_1slot_170/train',
	tfrecords_test = '/pqry/pqry/data/tfrecords_1slot_170/test',
	results = '/pqry/pqry/results',
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

