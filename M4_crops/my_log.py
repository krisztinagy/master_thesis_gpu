import config as cfg
import os
import shutil
import datetime

def training_header():

    directory = cfg.directory['results'] + '/' + cfg.model['model_import']
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    shutil.copy2(cfg.log['config_file'], directory)

    f = open(directory + '/log_train', 'a+')
    f.write('\n\n----TRAINING----\n\n')
    f.write('Batch size: %d\n' % (cfg.hyperparameters['batch_size']))
    f.write('Image size: %d * %d * %d\n' % (cfg.image['height'], cfg.image['width'], cfg.image['channels']))
    f.write('Model: %s\n' % (cfg.model['model_import']))
    f.write('Loss function: %s\n\n' % (cfg.model['loss_function_import']))
    f.write('Started: %s (GMT)\n\n' % str(datetime.datetime.now()))
    f.close()
    
    if not os.path.exists(os.path.join(directory, 'last_checkpoint')):
    
        g = open(directory + '/last_checkpoint', 'w')
        g.write('0')
        g.close()
        
    g = open(directory + '/last_checkpoint', 'r')
    last_checkpoint = int(g.readline())
    g.close()
    
    return directory, datetime.datetime.now(), last_checkpoint
    
def testing_header():
    
    directory = cfg.directory['results'] + '/' + cfg.model['model_import']
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    f = open(directory + '/log_test', 'a+')
    f.write('\n\n----TESTING----\n\n')
    f.write('Batch size: %d\n' % (cfg.hyperparameters['batch_size']))
    f.write('Image size: %d * %d * %d\n' % (cfg.image['height'], cfg.image['width'], cfg.image['channels']))
    f.write('Model: %s\n' % (cfg.model['model_import']))
    f.write('Loss function: %s\n\n' % (cfg.model['loss_function_import']))
    f.write('Started: %s (GMT)\n\n' % str(datetime.datetime.now()))
    f.close()
    
    return directory, datetime.datetime.now()
    
def logging_step(log_dir, step, loss, last_checkpoint):
    f = open(log_dir + '/log_train', 'a+')
    f.write('%s (GMT) Smoothed loss in current step %d (overall step %d): %.6f\n' % (str(datetime.datetime.now()), step, step+last_checkpoint, loss))
    f.close
    
    g = open(log_dir + '/last_checkpoint', 'w')
    g.write(str(last_checkpoint + step))
    g.close()
    
def logging_general(log_dir, file, string):
    f = open(log_dir + '/' + file, 'a+')
    f.write('%s\n' % string)
    f.close
    
def training_footer(log_dir, start_time):
    f = open(log_dir + '/log_train', 'a+')
    f.write('Finished: %s (GMT)\n' % ( str(datetime.datetime.now()) ) )
    f.write('Took: %s\n' % ( str(start_time - datetime.datetime.now()) ))
    f.write('\nTraining finished successfully\n')
    f.close
    
def testing_footer(log_dir, start_time):
    f = open(log_dir + '/log_test', 'a+')
    f.write('Finished: %s (GMT)\n' % ( str(datetime.datetime.now()) ) )
    f.write('Took: %s\n' % ( str(start_time - datetime.datetime.now()) ))
    f.write('\nTesting finished successfully\n')
    
    f.close
    
def loss_log(log_dir, iteration, step_loss, smoothed_loss):
    f = open(log_dir + '/losses', 'a+')
    f.write('%d,%.6f,%.6f\n' % (iteration, step_loss, smoothed_loss))
    f.close