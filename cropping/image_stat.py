#import tensorflow as tf
import sys
import os
from collections import Counter

def get_stat(dir, file):

    file.write( 'Occurrences in dictionary %s:\n' % dir )
    print('Determining number of files in %s.' % dir)
    
    filenames = os.listdir(dir)
    
    print('Found %d JPEG files inside %s.' % (len(filenames), dir))
    
    external_ids = []
    
    for i in filenames:
        id, _ = i.split('_')
        external_ids.append(id)
        
    counter_dict = Counter(external_ids)
    
    for dict_item in sorted(counter_dict):
        file.write( str(dict_item) + ': ' + str(counter_dict[dict_item]) + '\n' )
    
    return counter_dict.items()


######################################################
######################################################
################         MAIN          ###############
######################################################
######################################################
f = open('stat.txt', 'a+')

get_stat('/pqry/data/117_amneville_cropped/training/label0', f)
get_stat('/pqry/data/117_amneville_cropped/training/label1', f)
get_stat('/pqry/data/117_amneville_cropped/testing/label0', f)
get_stat('/pqry/data/117_amneville_cropped/testing/label1', f)

f.close()