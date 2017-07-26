image = dict(
	channels = 1,
)
dataset = dict(
    label0_folder = 'label0',
    label1_folder = 'label1',
	num_categories = 2,

 )
dir = dict(
    data = '/pqry/pqry/data',
    crops = '/pqry/pqry/data/170_cropped',
    full_images = '/pqry/pqry/data/170',
    tf_records = 'tfrecords_1slot_170',
)
files = dict(
    json = 'spaceek-maze-one.json',
    csv = 'telAvivLabels.csv'
)
processing = dict(
	training_shards = 100,
	testing_shards = 100,
    threads = 10,
)

threads = dict(
	preprocess = 1,
)