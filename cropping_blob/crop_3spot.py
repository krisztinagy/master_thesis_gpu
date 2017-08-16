import os
import json
import csv
from PIL import Image

import config_3spot as cfg
import my_log as mylog

import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('json_file_name', os.path.join( cfg.dir['data'], cfg.files['json'] ), 'Spot definition file')
flags.DEFINE_string('csv_file_name', os.path.join( cfg.dir['data'], cfg.files['csv'] ), 'Occupancy database')

flags.DEFINE_string('full_images_dir', os.path.join( cfg.dir['data'], cfg.dir['full_images'] ), 'Folder for storing full images')
flags.DEFINE_string('crops_dir', os.path.join( cfg.dir['data'], (cfg.dir['full_images'] + '_3slot_crops' + cfg.dir['crops_special_note']) ), 'Folder for storing crops')

flags.DEFINE_integer('num_classes', cfg.dataset['num_categories'], 'Number of classes')

flags.DEFINE_string('training_dates_list', cfg.dates['training'], 'Folder for label 1')
flags.DEFINE_string('testing_dates_list', cfg.dates['testing'], 'Folder for label 1')

def calculate_neighboring_slots(spot):

    # 170
    # return {
        # '11601': ('10400', '10401', '10402'),
        # '11602': ('10401', '10402', '10403'),
        # '11603': ('10402', '10403', '10404'),
        # '11604': ('10418', '10417', '10416'),
        # '11605': ('10419', '10418', '10417'),
        # '11606': ('10420', '10419', '10418'),
    # }[spot]
    
    # 49 
    return {
        '4901': ('4900', '4901', '4904'),
        '4904': ('4901', '4904', '4906'),
        '4906': ('4904', '4906', '4908'),
        '4908': ('4906', '4908', '4910'),
        '4903': ('4902', '4903', '4905'),
        '4905': ('4903', '4905', '4907'),
        '4907': ('4905', '4907', '4909'),
    }[spot]

def calculate_slot_coordinates(spots, spot):

    # slot = str(spots[str(spot)]["external_id"])
    # slot_before = str(int(spots[str(spot)]["external_id"]) - 1)
    # slot_after = str(int(spots[str(spot)]["external_id"]) + 1)
    
    slot_before, slot, slot_after = calculate_neighboring_slots(spot)
    
    slot_before_rot = spots[slot_before]['rotation']
    slot_before_x = spots[slot_before]['roi_rel']['x']
    slot_before_y = spots[slot_before]['roi_rel']['y']
    slot_before_height = spots[slot_before]['roi_rel']['height']
    slot_before_width = spots[slot_before]['roi_rel']['width']
    
    slot_rot = spots[slot]['rotation']
    slot_x = spots[slot]['roi_rel']['x']
    slot_y = spots[slot]['roi_rel']['y']
    slot_height = spots[slot]['roi_rel']['height']
    slot_width = spots[slot]['roi_rel']['width']
    
    slot_after_rot = spots[slot_after]['rotation']
    slot_after_x = spots[slot_after]['roi_rel']['x']
    slot_after_y = spots[slot_after]['roi_rel']['y']
    slot_after_height = spots[slot_after]['roi_rel']['height']
    slot_after_width = spots[slot_after]['roi_rel']['width']
    
    rot = sum([slot_before_rot, slot_rot, slot_after_rot]) / 3
    x = min(slot_before_x, slot_x, slot_after_x) + min(slot_before_width, slot_width, slot_after_width) / 4
    y = sum([slot_before_y, slot_y, slot_after_y]) / 3
    width = (slot_before_width + slot_width + slot_after_width) * 3 / 4
    height = max(slot_before_height, slot_height, slot_after_height) * 4 / 5
    
    return rot, x, y, width, height
    
def make_directories():

    if not os.path.exists(os.path.join( FLAGS.crops_dir, 'training/label000')):
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label000') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label001') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label010') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label011') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label100') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label101') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label110') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label111') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label000') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label001') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label010') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label011') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label100') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label101') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label110') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label111') )

def run_cropping():

    with open(FLAGS.json_file_name, 'r') as f:
        data = json.loads(f.read())
        
    with open(FLAGS.csv_file_name, 'rb') as csvfile:

        readCSV=csv.reader(csvfile, delimiter=',')
        labels=[]
        date_times=[]
        slot_ids=[]
        dates_and_slots=[]
        
        for row in readCSV:
            label=row[0]
            date_time=row[1]
            slot_id=row[2]
            
            if int(label) >= 1:
                labels.append(1)
            else:
                labels.append(int(label))
            
            date_string = date_time[0:4] + date_time[5:7] + date_time[8:10] + date_time[11:13] + date_time[14:16]
            date_times.append( date_string )
            slot_ids.append(slot_id)
            dates_and_slots.append( str(date_string) + str(slot_id) )
            
    print("Number of rows read from %s file:" % FLAGS.csv_file_name)
    print(len(labels))
    
    print(dates_and_slots[0])
    
    make_directories()

    labelled_crops = 0
    training_crops = 0
    testing_crops = 0

    # internal_slots = [11601, 11602, 11603, 11604, 11605, 11606]
    
    internal_slots = [4901, 4903, 4904, 4905, 4906, 4907, 4908]

    dates = sorted(os.listdir( FLAGS.full_images_dir ))
    print('Content of dir %s:' % ( FLAGS.full_images_dir ))
    print(dates)

    training_dates = FLAGS.training_dates_list.split()
    testing_dates = FLAGS.testing_dates_list.split()
    
    mylog.logging_general(FLAGS.crops_dir, '-----CROPPING TRAINING IMAGES-----')
    
    for date in training_dates:
    
        imagefiles = sorted(os.listdir(os.path.join( FLAGS.full_images_dir, date )))
        mylog.logging_general(FLAGS.crops_dir, 'Found %d images in dir %s.' % (len(imagefiles), str(date)))
        print('Found %d images in dir %s.' % (len(imagefiles), str(date)))
        
        decimal = 0
        for imagefile in imagefiles:
        
            decimal += 1
            if decimal % 1 == 0:
        
                orig_img = Image.open(os.path.join( FLAGS.full_images_dir, date, imagefile ))
                print('Processing image %s' % str(imagefile))
                
                for spot in data['spots']:
                
                    # Check if the spot is an internal spot - so that it has at least one left and one right neighbor
                    spot_nr = int( data['spots'][str(spot)]["external_id"] )
                    if  spot_nr in internal_slots:

                        rot, x, y, width, height = calculate_slot_coordinates(data['spots'], spot)
                        
                        date_time = imagefile[0:12]
                        
                        slot = str(data['spots'][str(spot)]["external_id"])
                        slot_before, slot, slot_after = calculate_neighboring_slots(slot)
                        
                        lookup_slot = date_time + slot
                        lookup_slot_before = date_time + slot_before
                        lookup_slot_after = date_time + slot_after
                        
                        if lookup_slot in dates_and_slots:
                            if lookup_slot_before in dates_and_slots:
                                if lookup_slot_after in dates_and_slots:
                                
                                    slot_label = labels[dates_and_slots.index(lookup_slot)]
                                    slot_label_before = labels[dates_and_slots.index(lookup_slot_before)]
                                    slot_label_after = labels[dates_and_slots.index(lookup_slot_after)]
                                    
                                    if slot_label != -1 and slot_label_before != -1 and slot_label_after != -1:
                                            
                                        labelled_crops += 1
                                        training_crops += 1
                                        if labelled_crops % 100 == 0:
                                            print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                                        
                                        aggregate_label = str(slot_label_before) + str(slot_label) + str(slot_label_after)
                                        folder_string = 'label' + str(aggregate_label) + '_folder'
                                        label = cfg.dataset[folder_string]
                                        
                                        img = orig_img.rotate(rot, expand=True)
                                        w, h = img.size
                                        left = int(x*w)
                                        top = int(y*h)
                                        right = left + int(width*w)
                                        bottom = top + int(height*h)                       
                                        img = img.crop((left, top, right, bottom))
                                        path = os.path.join( FLAGS.crops_dir, 'training', label, slot + '_' + imagefile)
                                
                                        img.save(path)

        mylog.logging_general(FLAGS.crops_dir, 'Number of training crops: %d' % (training_crops))
    
    mylog.logging_general(FLAGS.crops_dir, '-----CROPPING TESTING IMAGES-----')
                                        
    for date in testing_dates:
    
        imagefiles = sorted(os.listdir(os.path.join( FLAGS.full_images_dir, date )))
        mylog.logging_general(FLAGS.crops_dir, 'Found %d images in dir %s.' % (len(imagefiles), str(date)))
        print('Found %d images in dir %s.' % (len(imagefiles), str(date)))
        
        decimal = 0
        for imagefile in imagefiles:
        
            decimal += 1
            if decimal % 5 == 0:
        
                orig_img = Image.open(os.path.join( FLAGS.full_images_dir, date, imagefile ))
                print('Processing image %s' % str(imagefile))
                
                for spot in data['spots']:
                
                    # Check if the spot is an internal spot - so that it has at least one left and one right neighbor
                    spot_nr = int( data['spots'][str(spot)]["external_id"] )
                    if  spot_nr in internal_slots:

                        rot, x, y, width, height = calculate_slot_coordinates(data['spots'], spot)
                        
                        date_time = imagefile[0:12]
                        
                        slot = str(data['spots'][str(spot)]["external_id"])
                        slot_before, slot, slot_after = calculate_neighboring_slots(slot)
                        
                        lookup_slot = date_time + slot
                        lookup_slot_before = date_time + slot_before
                        lookup_slot_after = date_time + slot_after
                        
                        if lookup_slot in dates_and_slots:
                            if lookup_slot_before in dates_and_slots:
                                if lookup_slot_after in dates_and_slots:
                                
                                    slot_label = labels[dates_and_slots.index(lookup_slot)]
                                    slot_label_before = labels[dates_and_slots.index(lookup_slot_before)]
                                    slot_label_after = labels[dates_and_slots.index(lookup_slot_after)]
                                    
                                    if slot_label != -1 and slot_label_before != -1 and slot_label_after != -1:
                                            
                                        labelled_crops += 1
                                        testing_crops += 1
                                        if labelled_crops % 100 == 0:
                                            print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                                        
                                        aggregate_label = str(slot_label_before) + str(slot_label) + str(slot_label_after)
                                        folder_string = 'label' + str(aggregate_label) + '_folder'
                                        label = cfg.dataset[folder_string]
                                        
                                        img = orig_img.rotate(rot, expand=True)
                                        w, h = img.size
                                        left = int(x*w)
                                        top = int(y*h)
                                        right = left + int(width*w)
                                        bottom = top + int(height*h)                       
                                        img = img.crop((left, top, right, bottom))
                                        path = os.path.join( FLAGS.crops_dir, 'testing', label, slot + '_' + imagefile)
                                
                                        img.save(path)
                                        
        mylog.logging_general(FLAGS.crops_dir, 'Number of testing crops: %d' % (testing_crops))

    mylog.logging_general(FLAGS.crops_dir, 'All crops %d' % labelled_crops)
    mylog.logging_general(FLAGS.crops_dir, 'All training crops %d' % training_crops)
    mylog.logging_general(FLAGS.crops_dir, 'All testing crops: %d' % testing_crops)
        
    print('All crops %d' % labelled_crops)                    
    print('All training crops %d' % training_crops)
    print('All testing crops: %d' % testing_crops)

start_time = mylog.cropping_header(FLAGS.crops_dir, FLAGS.full_images_dir, FLAGS.crops_dir, FLAGS.json_file_name, FLAGS.csv_file_name )

run_cropping()

mylog.cropping_footer(FLAGS.crops_dir, start_time)