import os
import json
import csv
from PIL import Image

import config_1spot as cfg
import my_log as mylog

import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('json_file_name', os.path.join( cfg.dir['data'], cfg.files['json'] ), 'Spot definition file')
flags.DEFINE_string('csv_file_name', os.path.join( cfg.dir['data'], cfg.files['csv'] ), 'Occupancy database')

flags.DEFINE_string('full_images_dir', os.path.join( cfg.dir['data'], cfg.dir['full_images'] ), 'Folder for storing full images')
flags.DEFINE_string('crops_dir', os.path.join( cfg.dir['data'], (cfg.dir['full_images'] + '_1slot_crops' + cfg.dir['crops_special_note']) ), 'Folder for storing crops')

flags.DEFINE_integer('num_classes', cfg.dataset['num_categories'], 'Number of classes')
flags.DEFINE_string('label0', cfg.dataset['label0_folder'], 'Folder for label 0')
flags.DEFINE_string('label1', cfg.dataset['label1_folder'], 'Folder for label 1')

flags.DEFINE_string('training_dates_list', cfg.dates['training'], 'Folder for label 1')
flags.DEFINE_string('testing_dates_list', cfg.dates['testing'], 'Folder for label 1')


def run_cropping():

    with open(FLAGS.json_file_name, 'r') as f:
        data = json.loads(f.read())
        
    with open(FLAGS.csv_file_name, 'rb') as csvfile:

        readCSV=csv.reader(csvfile, delimiter=',')
        labels=[]
        date_times=[]
        slot_ids=[]
        
        for row in readCSV:
            label=row[0]
            date_time=row[1]
            slot_id=row[2]
            
            labels.append(int(label))
            date_times.append(date_time[0:4] + date_time[5:7] + date_time[8:10] + date_time[11:13] + date_time[14:16] + date_time[17:19])
            slot_ids.append(slot_id)
            
    print("Number of rows read from %s file:" % FLAGS.csv_file_name)
    print(len(labels))

    if not os.path.exists(os.path.join( FLAGS.crops_dir, 'training/label0')):
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label0') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label1') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label0') )
        os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label1') )

    labelled_crops = 0
    training_crops = 0
    training_crops_0 = 0
    training_crops_1 = 0
    testing_crops = 0
    testing_crops_0 = 0
    testing_crops_1 = 0

    dates = sorted(os.listdir( FLAGS.full_images_dir ))
    print('Content of dir %s:' % ( FLAGS.full_images_dir ) )
    print(dates)

    training_dates = FLAGS.training_dates_list.split()
    testing_dates = FLAGS.testing_dates_list.split()
    
    occluded_spots = ['10401', '10402', '10403', '10417', '10418', '10419']
    
    mylog.logging_general(FLAGS.crops_dir, '-----CROPPING TRAINING IMAGES-----')

    for date in training_dates:
    
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
                
                    if spot in occluded_spots:
                            
                        date_time = imagefile[0:14]
                        slot = str(data['spots'][str(spot)]["external_id"])

                        for i in range(len(date_times)):

                            if date_times[i] == date_time and slot_ids[i] == slot:
                            
                                if labels[i] != -1:
                                    
                                    labelled_crops += 1
                                    training_crops += 1
                                    if labelled_crops % 100 == 0:
                                        print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                                    
                                    if labels[i] == 0:
                                        label = cfg.dataset['label0_folder']
                                        training_crops_0 += 1
                                    else:
                                        label = cfg.dataset['label1_folder']
                                        training_crops_1 += 1
                                    
                                    img = orig_img.rotate(data['spots'][str(spot)]['rotation'], expand=True)
                                    w, h = img.size
                                    left = int(data['spots'][str(spot)]['roi_rel']['x']*w)
                                    top = int(data['spots'][str(spot)]['roi_rel']['y']*h)
                                    right = int(data['spots'][str(spot)]['roi_rel']['width']*w) + left
                                    bottom = int(data['spots'][str(spot)]['roi_rel']['height']*h) + top
                                    img = img.crop((left, top, right, bottom))
                                    path = os.path.join( FLAGS.crops_dir, 'training', label, slot + '_' + imagefile)
                            
                                    img.save(path)
        
        mylog.logging_general(FLAGS.crops_dir, 'Number of training crops: %d' % (training_crops))
        mylog.logging_general(FLAGS.crops_dir, 'Number of training crops with label 0: %d' % (training_crops_0))
        mylog.logging_general(FLAGS.crops_dir, 'Number of training crops with label 1: %d' % (training_crops_1))
    
    mylog.logging_general(FLAGS.crops_dir, '\n-----CROPPING TESTING IMAGES-----')
    
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
                
                    if spot in occluded_spots:
                            
                        date_time = imagefile[0:14]
                        slot = str(data['spots'][str(spot)]["external_id"])

                        for i in range(len(date_times)):

                            if date_times[i] == date_time and slot_ids[i] == slot:
                            
                                if labels[i] != -1:
                                    
                                    labelled_crops += 1
                                    testing_crops += 1
                                    if labelled_crops % 100 == 0:
                                        print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                                    
                                    if labels[i] == 0:
                                        label = cfg.dataset['label0_folder']
                                        testing_crops_0 += 1
                                    else:
                                        label = cfg.dataset['label1_folder']
                                        testing_crops_1 += 1
                                    
                                    img = orig_img.rotate(data['spots'][str(spot)]['rotation'], expand=True)
                                    w, h = img.size
                                    left = int(data['spots'][str(spot)]['roi_rel']['x']*w)
                                    top = int(data['spots'][str(spot)]['roi_rel']['y']*h)
                                    right = int(data['spots'][str(spot)]['roi_rel']['width']*w) + left
                                    bottom = int(data['spots'][str(spot)]['roi_rel']['height']*h) + top
                                    img = img.crop((left, top, right, bottom))
                                    path = os.path.join( FLAGS.crops_dir, 'testing', label, slot + '_' + imagefile)
                            
                                    img.save(path)
                                    
        mylog.logging_general(FLAGS.crops_dir, 'Number of testing crops: %d' % (testing_crops))
        mylog.logging_general(FLAGS.crops_dir, 'Number of testing crops with label 0: %d' % (testing_crops_0))
        mylog.logging_general(FLAGS.crops_dir, 'Number of testing crops with label 1: %d' % (testing_crops_1))
        
    mylog.logging_general(FLAGS.crops_dir, '\nAll crops %d' % labelled_crops)
    mylog.logging_general(FLAGS.crops_dir, 'All training crops %d' % training_crops)
    mylog.logging_general(FLAGS.crops_dir, 'All testing crops: %d' % testing_crops)        
                                    
    print('All crops %d' % labelled_crops)                    
    print('All training crops %d' % training_crops)
    print('All testing crops: %d' % testing_crops)
    
start_time = mylog.cropping_header(FLAGS.crops_dir, FLAGS.full_images_dir, FLAGS.crops_dir, FLAGS.json_file_name, FLAGS.csv_file_name )

run_cropping()

mylog.cropping_footer(FLAGS.crops_dir, start_time)
