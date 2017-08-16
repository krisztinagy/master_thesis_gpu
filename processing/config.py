# dataset = dict(
	# num_categories = 2,
    # label0 = 'label0',
    # label1 = 'label1',
    # percentage_to_use = 1,
# )

dataset = dict(
	num_categories = 8,
    label000_folder = 'label000',
    label001_folder = 'label001',
    label010_folder = 'label010',
    label011_folder = 'label011',
    label100_folder = 'label100',
    label101_folder = 'label101',
    label110_folder = 'label110',
    label111_folder = 'label111',
    percentage_to_use = 1,
)

image = dict(
	height = 32,
	width = 64,
	channels = 1,
)

model = dict(
	vgg19_pretrained = '/pqry/pqry/data/vgg19.npy',
    model_import = 'M3',
    model_dir_special_note = '',
    dataset = '3slot_cam170_occluded_5th_8classes',
	loss_function_import = 'cross_entropy',
)

directory = dict(
	results = '/pqry/pqry/thesis_results',
    tensorboard = 'tensorboard_logs',
)

fully_connected = dict(
    hidden = 1024,
    hidden1 = 1024,
    hidden2 = 1024,
)

hyperparameters = dict(
	learning_rate = 0.01,
	batch_size = 50,
	num_epochs = 50,
	num_epochs_eval = 1,
    num_threads = 10,
)

testing = dict(
    crop_percentage = 1,
)

log = dict(
    config_file = 'config.py',
)



