dataset = dict(
	num_categories = 2,
    label000_folder = 'label000',
    label001_folder = 'label001',
    label010_folder = 'label010',
    label011_folder = 'label011',
    label100_folder = 'label100',
    label101_folder = 'label101',
    label110_folder = 'label110',
    label111_folder = 'label111',
    divider = 1,
)

dir = dict(
    data = '/pqry/pqry/data',
      
    # 170 occluded
    full_images = '170',
    crops_special_note = '_occluded_all',
    records_special_note = '_occluded_all_2classes',
    
    # 49
    # full_images = '49',
    # crops_special_note = '_all',
    # records_special_note = '_all_8classes',    
)

files = dict(
    # 170
    json = 'spaceek-116.json',
    csv = 'telAvivLabels.csv'
    
    # 117
    # json = 'mentis-amneville-one.json',
    # csv = 'occupancy117.csv'

    # 115
    # json = 't2-hq.json',
    # csv = 'occupancy115.csv'
    
    # # 29
    # json = 'locarno-29.json',
    # csv = 'cam29.csv'
    
    # # 49
    # json = 'locarno-49.json',
    # csv = 'cam49.csv'
)

dates = dict(
    # 170
    training = '170301 170303 170306 170308 170310 170313 170315 170317 170320 170322 170324 170327 170329 170331 170323 170328 170330',
    testing = '170302 170307 170309 170314 170316 170321',
    
    #117
    # training = '170324 170325 170327 170328 170329 170331',
    # testing = '170326 170330',   
    
    # 115
    # training = '170306 170308 170310 170311 170313 170314 170315',
    # testing = '170307 170309 170312',
    
    # # 29
    # training = 
    # # '150121 150122 150123 150126 150128 150130 '
    # # '150202 150203 150204 150206 150209 '
    # # '150401 150403 150406 150407 150408 150410 '
    # # '150501 150504 150506 150508 150511 150513 150514 150515 150518 150519 150520 150522 150525 150527 150529',
    # '150501 150504 150506 150508 150511 150513 150514 150515 ',
    # testing = 
    # # '150127 150129 '
    # # '150205 150210 '
    # # '150402 150409 '
    # # '150505 150507 150512 150521 150526 150528',
    # '150505 150507 150512',
    
    # # 49
    # training = '150424 150427 150429 '
    # '150501 150504 150506 150508 150511 150513 150515 150518 150520 ' +
    # '150601 150603 150605 150608 150610 150612 ' +
    # '150701 150703 150708 150710',
    # testing = '150423 150428 150430 '
    # '150505 150507 150512 150514 150519 ' +
    # '150602 150604 150609 150611 ' +
    # '150702 150709',
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