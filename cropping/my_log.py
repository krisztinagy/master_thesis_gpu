import config as cfg
import os
import shutil
import datetime

def cropping_header(directory, full_images, crops, json, csv):

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    shutil.copy2('config.py', directory)

    f = open(directory + '/log', 'a+')
    
    f.write('Full images: %s\n' % (full_images))
    f.write('Crops: %s\n' % (crops))
    f.write('JSON file: %s\n' % (json))
    f.write('CSV file: %s\n' % (csv))
    
    f.write('Started: %s (GMT)\n\n' % str(datetime.datetime.now()))
    
    f.close()
    
    return datetime.datetime.now()
    
def logging_general(log_dir, string):
    f = open(log_dir + '/log', 'a+')
    f.write('%s\n' % string)
    f.close
    
def cropping_footer(directory, start_time):
    
    f = open(directory + '/log', 'a+')
    
    f.write('Finished: %s (GMT)\n' % ( str(datetime.datetime.now()) ) )
    f.write('Took: %s\n' % ( str(datetime.datetime.now() - start_time) ))
    f.write('\nCropping finished successfully\n')
    
    f.close