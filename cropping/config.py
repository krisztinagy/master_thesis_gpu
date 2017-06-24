image = dict(
	channels = 1,
)
dataset = dict(
    label0_folder = 'label0',
    label1_folder = 'label1',
	num_categories = 2,

 )
dir = dict(
    data = '/pqry/data',
    crops = '/pqry/data/117_amneville_cropped',
    full_images = '/pqry/data/117',
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