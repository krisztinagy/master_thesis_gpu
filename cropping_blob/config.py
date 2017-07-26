image = dict(
	channels = 1,
)
dataset = dict(
    label000_folder = 'label000',
    label001_folder = 'label001',
    label010_folder = 'label010',
    label011_folder = 'label011',
    label100_folder = 'label100',
    label101_folder = 'label101',
    label110_folder = 'label110',
    label111_folder = 'label111',
	num_categories = 8,

 )
dir = dict(
    data = '/pqry/pqry/data',
    crops = '/pqry/pqry/data/29_locarno_cropped_3spot_8classes',
    full_images = '/pqry/pqry/data/29',
    tf_records = 'tfrecords_3slots_8classes_cam29',
)
files = dict(
    json = 'locarno-29.json',
    csv = 'cam29.csv'
)
processing = dict(
	training_shards = 100,
	testing_shards = 100,
    threads = 10,
)

threads = dict(
	preprocess = 1,
)