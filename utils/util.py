import os
import torch.nn as nn
import shutil
import torch.distributed as dist

def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def replace_layer(state_dict, keyword1, keyword2):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword1 in key:
            new_key = key.replace(keyword1, keyword2)
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def copy_dir(dir1, dir2):
    dlist = os.listdir(dir1)
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    for f in dlist:
        file1 = os.path.join(dir1, f)   
        file2 = os.path.join(dir2, f)   
        if os.path.isfile(file1):
            shutil.copyfile(file1, file2) 

def seek_class(args, path):
    if args.root == 'CUB_200_2011':
        val_root = 'Data/'+args.root + '/test'
        path_split = path.split("_")[:-2]
        part_path = ''
        for i in path_split:
            part_path += i + '_'
        part_path = part_path[:-1]
        part_path = part_path.lower()
        classes_name = os.listdir(val_root)
        for cur_class in classes_name:
            cur_class_low = cur_class.lower()
            if cur_class_low[4:] == part_path:
                path = 'Data/' + args.root + '/test/' + cur_class + '/' + path
                return path, int(cur_class[:3]) - 1, cur_class

    elif args.root == 'fgvc_aircraft_2013b':
        label_classes = {} 
        with open('Data/fgvc_aircraft_2013b/classes.txt', 'r') as f:
            for i, line in enumerate(f):
                label_classes[line[:-1]] = i
        with open('Data/fgvc_aircraft_2013b/images_box.txt', 'r') as f:
            for line in f:
                img_name = line.split(';')[0]
                if img_name == path[:-4]:
                    img_label = line.split(';')[1]
                    path = 'Data/fgvc_aircraft_2013b/images' + '/' + path
                    return path, label_classes[img_label], img_label
                
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def copy_dir(dir1, dir2):
    dlist = os.listdir(dir1)
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    for f in dlist:
        file1 = os.path.join(dir1, f)  
        file2 = os.path.join(dir2, f)    
        if os.path.isfile(file1):
            shutil.copyfile(file1, file2) 

def seek_class(args, path):
    if args.root == 'CUB_200_2011':
        val_root = '/home2/Datasets/' + args.root + '/test'
        path_split = path.split("_")[:-2]
        part_path = ''
        for i in path_split:
            part_path += i + '_'
        part_path = part_path[:-1]
        part_path = part_path.lower()
        classes_name = os.listdir(val_root)
        for cur_class in classes_name:
            cur_class_low = cur_class.lower()
            if cur_class_low[4:] == part_path:
                path = val_root + '/' + cur_class + '/' + path
                return path, int(cur_class[:3]) - 1, cur_class

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0