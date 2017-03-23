import os,sys
import config

def mkdir(path):
    # make directory recursive, doesn't have warry info when path exists
    os.system('mkdir -p {}'.format(path))


def log(log_file,log_content):
    # write log information
    mkdir(config.log_folder)
    log_file_path = os.path.join(config.log_folder,log_file)
    with open(log_file_path,'a') as fout:
        fout.write(log_content+'\n')
