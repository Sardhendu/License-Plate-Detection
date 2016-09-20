
from BldModel import SvmModel, Models
import Configuration
import Bld_FeatureCrps
from Classify import Classify
import numpy as np

from Test import Test

#==============================================================================
# 1: create and store features and labels of the images: (use Bld_FeatureCrps)
#==============================================================================
features, labels = Bld_FeatureCrps().create_training_feature(store = 'y')


#==============================================================================
# 2: Build model using the training feature and store the models int he disk
#==============================================================================
'''
    MODEL 1: find the Kernel cnvrt features, cost_function and Parameters (thetas) and store them as models
	Note:
		The more the sigma is the less the gamma will be
'''
sigma=4
max_iter=5000
alpha=0.003
c=10
features_krnl_cnvrt,j_theta, theta = SvmModel().main_call(sigma, features, labels, max_iter, alpha, c)
np.savetxt(conf["Data_feature_KernelCnvtd_dir"], features_krnl_cnvrt, delimiter=",")
np.savetxt(conf["Theta_val_dir"], theta, delimiter=",")
'''
    MODEL 2: Use packages models, linearSvc, and rbf's with varying gamma and c values.
'''
Models(type = 'rbf').fit(features)


#==============================================================================
# 3: Test Models and print the prediction output of the crossvalidation set:
#==============================================================================
Test().create_test_features(model = 'model_s')  # chooose a model model_s or models


####################
#==============================================================================
# 4: Find the Number plates of the vehicle
#==============================================================================
image_to_classify = "../image_2_classify (11).jpg"


def test_models():
	conf = Configuration.get_datamodel_storage_path()
	for files in glob.glob(conf['All_Models']):         # Test Using all the models saved in the disk
	    classifier= open(files).read()
	    classifier=cPickle.loads(classifier)
	    
	    image_to_classify = cv2.imread(image_to_classify_path)
	    image_resized = imutils.resize(image_to_classify, height=500)
	    pred = Classify().classify_new_instance(image_resized,classifier)
	    
	    print classifier
	    for i in range(0,len(pred)):
	        if pred[i][0]==1:
	            print pred[i]
