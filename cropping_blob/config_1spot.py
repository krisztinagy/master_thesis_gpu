dataset = dict(
	num_categories = 2,
    label0_folder = 'label0',
    label1_folder = 'label1',
    percentage_to_use = 1,
)

dir = dict(
    data = '/pqry/pqry/data',
    full_images = '170',
    crops_special_note = '_occluded_5th',
    records_special_note = '_occluded_5th',
)

files = dict(
    json = 'spaceek-maze-one.json',
    csv = 'telAvivLabels.csv'
)

dates = dict(
    training = '170301 170303 170306 170308 170310 170313 170315 170317 170320 170322 170324 170327 170329 170331',
    testing = '170302 170307 170309 170314 170316 170321 170323 170328 170330',
)

image = dict(
	channels = 1,
)

processing = dict(
	training_shards = 100,
	testing_shards = 100,
    threads = 10,
)

threads = dict(
	preprocess = 1,
)