import os
import json
import csv
from PIL import Image

import config as cfg
import random

def calculate_neighboring_slots(spot):
    return {
        '165': ('164', '166'),
        '170': ('167', '171'),
        '171': ('170', '172'),
        '172': ('171', '173'),
        '173': ('172', '174'),
        '174': ('173', '175'),
        '175': ('174', '176'),
        '176': ('175', '177'),
    }[spot]
    
    # return {
        # '4901': ('4900', '4904'),
        # '4904': ('4901', '4906'),
        # '4906': ('4904', '4908'),
        # '4908': ('4906', '4910'),
        # '4903': ('4902', '4905'),
        # '4905': ('4903', '4907'),
        # '4907': ('4905', '4909'),
    # }[spot]

def calculate_slot_coordinates(spots, spot):

    # slot = str(spots[str(spot)]["external_id"])
    # slot_before = str(int(spots[str(spot)]["external_id"]) - 1)
    # slot_after = str(int(spots[str(spot)]["external_id"]) + 1)
    
    slot = str(spots[str(spot)]["external_id"])
    slot_before, slot_after = calculate_neighboring_slots(slot)
    
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
    # x = slot_after_x
    x = slot_after_x + (slot_x - slot_after_x) / 3
    #y = sum([slot_before_y, slot_y, slot_after_y]) / 3
    y = slot_after_y
    #width = slot_before_width + slot_width + slot_after_width
    #width = (slot_before_x - slot_after_x) / 2 * 3 + ( slot_before_width / 2 )
    #width = (slot_before_x - slot_after_x) / 2 * 3
    right_most_pixel = slot_before_x +slot_before_width
    width = right_most_pixel - x 
    height = slot_before_height + (slot_y - slot_after_y)
    
    return rot, x, y, width, height
    
def make_directories():

    if not os.path.exists(cfg.dir['crops']):
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label000') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label001') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label010') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label011') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label100') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label101') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label110') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'training/label111') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label000') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label001') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label010') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label011') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label100') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label101') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label110') )
        os.makedirs(os.path.join( cfg.dir['crops'], 'testing/label111') )

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
        dates_and_slots=[]
        
        for row in readCSV:
            label=row[0]
            date_time=row[1]
            slot_id=row[2]
            
            if int(label) >= 1:
                labels.append(1)
            else:
                labels.append(0)
            
            date_string = date_time[0:4] + date_time[5:7] + date_time[8:10] + date_time[11:13] + date_time[14:16]
            date_times.append( date_string )
            slot_ids.append(slot_id)
            dates_and_slots.append( str(date_string) + str(slot_id) )
            
    print("Number of rows read from %s file:" % csv_file_name)
    print(len(labels))
    
    print(dates_and_slots[0])
    
    make_directories()

    labelled_crops = 0
    training_crops = 0
    testing_crops = 0

    #internal_slots = [4901, 4903, 4904, 4905, 4906, 4907, 4908]
    internal_slots = [170, 171, 172, 173, 174, 175, 176]

    dates = sorted(os.listdir( cfg.dir['full_images'] ))
    print('Content of dir %s:' % (cfg.dir['full_images']) )

    # dates = ['150607', '150608', '150609', '150610',
    #  '150611', '150612', '150613', '150614', '150601', '150701', '150702', '150703', '150708', '150709', '150710']
    dates = ['150501', '150502', '150503', '150504', '150505', '150506', '150507', '150508', '150509', '150510',
    '150511', '150512', '150513', '150514', '150515', '150516', '150517', '150518', '150519', '150520',
    '150521', '150522', '150523', '150524', '150525', '150526', '150527', '150528', '150529', '150530',
    '150531']
    # print(dates)
    
    for date in dates:
        imagefiles = sorted(os.listdir(os.path.join( cfg.dir['full_images'], date )))
        print('Found %d images in dir %s.' % (len(imagefiles), str(date)))
        for imagefile in imagefiles:
            orig_img = Image.open(os.path.join( cfg.dir['full_images'], date, imagefile ))
            print('Processing image %s' % str(imagefile))
            for spot in data['spots']:
            
                # Check if the spot is an internal spot - so that it has at least one left and one right neighbor
                spot_nr = int( data['spots'][str(spot)]["external_id"] )
                #print(spot_nr)
                if  spot_nr in internal_slots:
                    
                    rand = random.random()
                    
                    if rand < 0.8:
                        train_or_test = os.path.join(cfg.dir['crops'], 'training')
                        training_crops += 1
                    else:
                        train_or_test = os.path.join(cfg.dir['crops'], 'testing')
                        testing_crops += 1

                    rot, x, y, width, height = calculate_slot_coordinates(data['spots'], spot)
                    
                    date_time = imagefile[0:12]
                    slot = str(data['spots'][str(spot)]["external_id"])
                    slot_before, slot_after = calculate_neighboring_slots(slot)
                    
                    # slot_before = str(int(data['spots'][str(spot)]["external_id"]) - 1 )
                    # slot_after = str(int(data['spots'][str(spot)]["external_id"]) + 1 )
                    
                    lookup_slot = date_time + slot
                    lookup_slot_before = date_time + slot_before
                    lookup_slot_after = date_time + slot_after
                    
                    if lookup_slot in dates_and_slots:
                        if lookup_slot_before in dates_and_slots:
                            if lookup_slot_after in dates_and_slots:
                            
                                #print(1)
                                slot_index = dates_and_slots.index(lookup_slot)
                                slot_index_before = dates_and_slots.index(lookup_slot_before)
                                slot_index_after = dates_and_slots.index(lookup_slot_after)

                                labelled_crops += 1
                                if labelled_crops % 100 == 0:
                                    print("Number of labelled crops: %d, Slot external id: %s" % (labelled_crops, slot))
                                    
                                aggregate_label = str(labels[slot_index_before]) + str(labels[slot_index]) + str(labels[slot_index_after])
                                folder_string = 'label' + str(aggregate_label) + '_folder'
                                label = cfg.dataset[folder_string]
                                
                                #img = orig_img.rotate(data['spots'][str(spot)]['rotation'], expand=True)
                                img = orig_img.rotate(rot, expand=True)
                                w, h = img.size
                                left = int(x*w)
                                top = int(y*h)
                                right = left + int(width*w)
                                bottom = top + int(height*h)                       
                                img = img.crop((left, top, right, bottom))
                                path = os.path.join( train_or_test, label, slot + '_' + imagefile)
                                #print(path)
                        
                                img.save(path)

    print('All crops %d' % labelled_crops)                    
    print('All training crops %d' % training_crops)
    print('All testing crops: %d' % testing_crops)

main()