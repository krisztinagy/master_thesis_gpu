import os
import json
import csv
from PIL import Image

import config as cfg
import random

def calculate_slot_coordinates(spots, spot):

    slot = str(spots[str(spot)]["external_id"])
    slot_before = str(int(spots[str(spot)]["external_id"]) - 1)
    slot_after = str(int(spots[str(spot)]["external_id"]) + 1)
    
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
    x = min(slot_before_x, slot_x, slot_after_x)
    y = sum([slot_before_y, slot_y, slot_after_y]) / 3
    width = slot_before_width + slot_width + slot_after_width
    height = max(slot_before_height, slot_height, slot_after_height)
    
    return rot, x, y, width, height
    
def make_directories():

    if not os.path.exists(cfg.dir['crops']):
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label0') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label1') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label0') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label1') )

def main():

    json_file_name = os.path.join( cfg.dir['data'], cfg.files['json'] )
    csv_file_name = os.path.join( cfg.dir['data'], cfg.files['csv'] )

    with open(json_file_name, 'r') as f:
        data = json.loads(f.read())
        
    with open(csv_file_name, 'rb') as csvfile:

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
            
    print("Number of rows read from %s file:" % csv_file_name)
    print(len(labels))
    
    make_directories()

    labelled_crops = 0
    training_crops = 0
    testing_crops = 0

    internal_slots = [1, 2, 3, 4, 5, 6, 7, 8,
                      11, 12, 13,
                      16, 17, 18, 19, 20,
                      24, 25, 26, 27, 28, 29,
                      32, 33, 34, 35, 36, 37, 38,
                      41, 42, 43, 44, 45, 46, 47,
                      50, 51, 52, 53, 54, 55, 56,
                      59, 60, 61, 62, 63, 64, 65,
                      68, 69, 70, 71, 72, 73, 74,
                      77, 78, 79, 80, 81, 82, 83,
                      86, 87, 88, 89, 90, 91, 92,
                      95, 96, 97, 98, 99, 100, 101,
                      126, 127, 128, 129, 130, 131, 132, 133, 134,
                      138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154]

    dates = sorted(os.listdir( cfg.dir['full_images'] ))
    print('Content of dir %s:' % (cfg.dir['full_images']) )
    print(dates)
    for date in dates:
        imagefiles = sorted(os.listdir(os.path.join( cfg.dir['full_images'], date )))
        print('Found %d images in dir %s.' % (len(imagefiles), str(date)))
        for imagefile in imagefiles:
            orig_img = Image.open(os.path.join( cfg.dir['full_images'], date, imagefile ))
            for spot in data['spots']:
            
                # Check if the spot is an internal spot - so that it has at least one left and one right neighbor
                spot_nr = int( data['spots'][str(spot)]["external_id"] ) - 11700
                #print(spot_nr)
                if  spot_nr in internal_slots:
                    
                    rand = random.random()
                    
                    if rand < 0.8:
                        train_or_test = os.path.join(cfg.dir['crops'], 'training')
                        training_crops += 1
                    else:
                        train_or_test = os.path.join(cfg.dir['crops'], 'testing')
                        testing_crops += 1
                        
                    date_time = imagefile[0:14]
                    
                    rot, x, y, width, height = calculate_slot_coordinates(data['spots'], spot)
                    
                    slot = str(data['spots'][str(spot)]["external_id"])

                    for i in range(len(date_times)):
                        if date_times[i] == date_time and slot_ids[i] == slot:
                            
                            labelled_crops += 1
                            if labelled_crops % 100 == 0:
                                print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                            
                            if labels[i] == 0:
                                label = cfg.dataset['label0_folder']
                            else:
                                label = cfg.dataset['label1_folder']
                            
                            #img = orig_img.rotate(data['spots'][str(spot)]['rotation'], expand=True)
                            img = orig_img.rotate(rot, expand=True)
                            w, h = img.size
                            left = int(x*w)
                            top = int(y*h)
                            right = left + int(width*w)
                            bottom = top + int(height*h)                       
                            img = img.crop((left, top, right, bottom))
                            path = os.path.join( train_or_test, label, slot + '_' + imagefile)
                    
                            img.save(path)

    print('All crops %d' % labelled_crops)                    
    print('All training crops %d' % training_crops)
    print('All testing crops: %d' % testing_crops)

    
main()