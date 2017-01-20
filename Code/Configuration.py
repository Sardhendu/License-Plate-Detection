"""
Created on Tue Jan  5 13:35:56 2016

@author: sardendhu
"""

import os
import configparser as ConfigParser


def get_config_dir():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.abspath(os.path.join(dir_name, os.pardir))

    conf_name = 'config-local.conf'    
    dir_name = os.path.join(dir_name, conf_name)
    # print 'The directory where the config file is: ', dir_name
    return dir_name
    

def get_config():
    Config = ConfigParser.ConfigParser()   
    Config.read(get_config_dir())   
    print ('All the configuration are: ', Config.sections())
    return Config


def get_datamodel_storage_path(): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    curr_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
    curr_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
    curr_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
    print (curr_dir)

    conf = get_config()

    config_settings = {}
    config_settings["Train_data1_dir"] = curr_dir+'/'+conf.get("Dataset", "Train_data1")
    config_settings["Train_data2_dir"] = curr_dir+'/'+conf.get("Dataset", "Train_data2")
    config_settings["Valid_data1_dir"] = curr_dir+'/'+conf.get("Dataset", "Valid_data1")
    config_settings["Valid_data2_dir"] = curr_dir+'/'+conf.get("Dataset", "Valid_data2")

    config_settings["Data_feature_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Data_feature")
    config_settings["Class_labels_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Class_labels")
    config_settings["Data_feature_KernelCnvtd_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Data_feature_KernelCnvtd")
    config_settings["Data_feature_KernelCnvtd_tst_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Data_feature_KernelCnvtd_tst")
    config_settings["Theta_val_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Theta_val")

    config_settings["Linear_SVC_dir"] = curr_dir+'/'+conf.get("Models", "Linear_SVC")
    config_settings["SVM_RFB_dir"] = curr_dir+'/'+conf.get("Models","SVM_RFB")+'model_svm_rbf_%i_%i.pickle'
    config_settings["Models"] = curr_dir+'/'+conf.get("Models", "models")

    config_settings["Regions_of_Intrest"] = curr_dir+'/'+conf.get("Contored_images", "Regions_of_Intrest")

    config_settings["Indian_cars"] = curr_dir+'/'+conf.get("Images_to_classify","Indian_cars")
    config_settings["Foreign_cars"] = curr_dir+'/'+conf.get("Images_to_classify","Foreign_cars")
   
    return config_settings


# print (get_config_dir())
# print (get_datamodel_storage_path())