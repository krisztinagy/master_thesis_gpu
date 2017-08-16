import os
import json
import csv

import config as cfg
import my_log as mylog

def run_stat():

    directory = cfg.directory['results'] + '/' + cfg.model['model_import'] + '_' + cfg.model['dataset'] + cfg.model['model_dir_special_note'] 
    
    filenames = []
    spots = []
    times = []
    labels = []
    predictions = []

    with open(directory + '/test_results', 'r') as csvfile:
        readCSV=csv.reader(csvfile, delimiter=',')

        for row in readCSV:
            file_path = row[0]
            filename = file_path.split('/')[7]
            spot = filename.split('_')[0]
            rest = filename.split('_')[1]
            time = rest.split('.')[0]
            
            label = row[1]
            prediction = row[2]
            
            filenames.append(str(filename))
            spots.append(int(spot))
            times.append(int(time))
            labels.append(int(label))
            predictions.append(int(prediction))
        
    internal_spots = [10402, 10417, 10418]
    
    correct_predictions = 0
    incorrect_predictions = 0
    
    print(len(filenames))
        
    for i in range( len(filenames) ):
        
        if spots[i] in internal_spots:
            
            this_spot = spots[i]
            this_prediction3 = predictions[i]
            if this_prediction3 == 0 or this_prediction3 == 1 or this_prediction3 == 4 or this_prediction3 == 5:
                this_prediction1 = 0
            else:
                this_prediction1 = 1
            
            for j in range( len(filenames) ):
                
                if this_spot == spots[j] + 1 and times[i] == times[j]:
                    
                    prediction3_left = predictions[j]
                    
                    if prediction3_left == 0 or prediction3_left == 2 or prediction3_left == 4 or prediction3_left == 6:
                        left_prediction1 = 0
                    else:
                        left_prediction1 = 1
                    
                if this_spot == spots[j] - 1 and times[i] == times[j]:
                
                    prediction3_right = predictions[j]
                    
                    if prediction3_right == 0 or prediction3_right == 1 or prediction3_right == 2 or prediction3_right == 3:
                        right_prediction1 = 0
                    else:
                        right_prediction1 = 1
            
            if this_prediction1 * 2 + left_prediction1 + right_prediction1 >= 3:
                overall_prediction = 1
            else:
                overall_prediction = 0
            
            if (this_prediction3 == 0 or this_prediction3 == 1 or this_prediction3 == 4 or this_prediction3 == 5) and overall_prediction == 0:
                correct_predictions += 1
            elif (this_prediction3 == 2 or this_prediction3 == 3 or this_prediction3 == 6 or this_prediction3 == 7) and overall_prediction == 1:
                correct_predictions += 1
            else:
                incorrect_predictions += 1
                
    print('Correct predictions: %d\n' % correct_predictions)
    print('Incorrect predictions: %d\n' % incorrect_predictions)
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%

    # if not os.path.exists(os.path.join( FLAGS.crops_dir, 'training/label0')):
        # os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label0') )
        # os.makedirs(os.path.join( FLAGS.crops_dir, 'training/label1') )
        # os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label0') )
        # os.makedirs(os.path.join( FLAGS.crops_dir, 'testing/label1') )

    # labelled_crops = 0
    # training_crops = 0
    # training_crops_0 = 0
    # training_crops_1 = 0
    # testing_crops = 0
    # testing_crops_0 = 0
    # testing_crops_1 = 0

    # dates = sorted(os.listdir( FLAGS.full_images_dir ))
    # print('Content of dir %s:' % ( FLAGS.full_images_dir ))
    # print(dates)

    # training_dates = FLAGS.training_dates_list.split()
    # testing_dates = FLAGS.testing_dates_list.split()
    
    # mylog.logging_general(FLAGS.crops_dir, '-----CROPPING TRAINING IMAGES-----')

    # for date in training_dates:
    
        # imagefiles = sorted(os.listdir(os.path.join( FLAGS.full_images_dir, date )))
        # mylog.logging_general(FLAGS.crops_dir, 'Found %d images in dir %s.' % (len(imagefiles), str(date)))
        # print('Found %d images in dir %s.' % (len(imagefiles), str(date)))
        
        # decimal  = 0
        # for imagefile in imagefiles:
        
            # decimal += 1
            # if decimal % FLAGS.divider == 0:
        
                # orig_img = Image.open(os.path.join( FLAGS.full_images_dir, date, imagefile ))
                # print('Processing image %s' % str(imagefile))
                
                # for spot in data['spots']:
                        
                    # date_time = imagefile[0:14]
                    # slot = str(data['spots'][str(spot)]["external_id"])

                    # for i in range(len(date_times)):

                        # if date_times[i] == date_time and slot_ids[i] == slot:
                            
                            # if labels[i] != -1:
                                
                                # labelled_crops += 1
                                # training_crops += 1
                                # if labelled_crops % 100 == 0:
                                    # print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                                
                                # if labels[i] == 0:
                                    # label = cfg.dataset['label0_folder']
                                    # training_crops_0 += 1
                                # else:
                                    # label = cfg.dataset['label1_folder']
                                    # training_crops_1 += 1
                                
                                # img = orig_img.rotate(data['spots'][str(spot)]['rotation'], expand=True)
                                # w, h = img.size
                                # left = int(data['spots'][str(spot)]['roi_rel']['x']*w)
                                # top = int(data['spots'][str(spot)]['roi_rel']['y']*h)
                                # right = int(data['spots'][str(spot)]['roi_rel']['width']*w) + left
                                # bottom = int(data['spots'][str(spot)]['roi_rel']['height']*h) + top
                                # img = img.crop((left, top, right, bottom))
                                # path = os.path.join( os.path.join(FLAGS.crops_dir), 'training', label, slot + '_' + imagefile)
                        
                                # img.save(path)

    
        # mylog.logging_general(FLAGS.crops_dir, 'Number of training crops: %d' % (training_crops))
        # mylog.logging_general(FLAGS.crops_dir, 'Number of training crops with label 0: %d' % (training_crops_0))
        # mylog.logging_general(FLAGS.crops_dir, 'Number of training crops with label 1: %d' % (training_crops_1))
    
    # mylog.logging_general(FLAGS.crops_dir, '-----CROPPING TESTING IMAGES-----')

    # for date in testing_dates:
    
        # imagefiles = sorted(os.listdir(os.path.join( FLAGS.full_images_dir, date )))
        # mylog.logging_general(FLAGS.crops_dir, 'Found %d images in dir %s.' % (len(imagefiles), str(date)))
        # print('Found %d images in dir %s.' % (len(imagefiles), str(date)))
        
        # decimal  = 0
        # for imagefile in imagefiles:
        
            # decimal += 1
            # if decimal % FLAGS.divider == 0:
        
                # orig_img = Image.open(os.path.join( FLAGS.full_images_dir, date, imagefile ))
                # print('Processing image %s' % str(imagefile))
                # for spot in data['spots']:
                        
                    # date_time = imagefile[0:14]
                    # slot = str(data['spots'][str(spot)]["external_id"])

                    # for i in range(len(date_times)):

                        # if date_times[i] == date_time and slot_ids[i] == slot:
                            
                            # if labels[i] != -1:
                                
                                # labelled_crops += 1
                                # testing_crops += 1
                                # if labelled_crops % 100 == 0:
                                    # print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                                
                                # if labels[i] == 0:
                                    # label = cfg.dataset['label0_folder']
                                    # testing_crops_0 += 1
                                # else:
                                    # label = cfg.dataset['label1_folder']
                                    # testing_crops_1 += 1
                                
                                # img = orig_img.rotate(data['spots'][str(spot)]['rotation'], expand=True)
                                # w, h = img.size
                                # left = int(data['spots'][str(spot)]['roi_rel']['x']*w)
                                # top = int(data['spots'][str(spot)]['roi_rel']['y']*h)
                                # right = int(data['spots'][str(spot)]['roi_rel']['width']*w) + left
                                # bottom = int(data['spots'][str(spot)]['roi_rel']['height']*h) + top
                                # img = img.crop((left, top, right, bottom))
                                # path = os.path.join( os.path.join(FLAGS.crops_dir), 'testing', label, slot + '_' + imagefile)
                        
                                # img.save(path)
                            
        # mylog.logging_general(FLAGS.crops_dir, 'Number of testing crops: %d' % (testing_crops))
        # mylog.logging_general(FLAGS.crops_dir, 'Number of testing crops with label 0: %d' % (testing_crops_0))
        # mylog.logging_general(FLAGS.crops_dir, 'Number of testing crops with label 1: %d' % (testing_crops_1))                            
    
    # mylog.logging_general(FLAGS.crops_dir, 'All crops %d' % labelled_crops)
    # mylog.logging_general(FLAGS.crops_dir, 'All training crops %d' % training_crops)
    # mylog.logging_general(FLAGS.crops_dir, 'All testing crops: %d' % testing_crops)
    
    # print('All crops %d' % labelled_crops)                    
    # print('All training crops %d' % training_crops)
    # print('All testing crops: %d' % testing_crops)

# start_time = mylog.cropping_header(FLAGS.crops_dir, FLAGS.full_images_dir, FLAGS.crops_dir, FLAGS.json_file_name, FLAGS.csv_file_name )

run_stat()

# mylog.cropping_footer(FLAGS.crops_dir, start_time)