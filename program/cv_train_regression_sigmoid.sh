#

python program/train_regression_sigmoid.py -g 1 --no-show -r 'resource/images' --initmodel resource/alex_sigmoid.model resource/cv_lists 1 


python program/estimate_regression.py -g 1 -init resource/cv_regression_sigmoid_model/remove_1/model -m resource/cv_lists/remove_1_mean.npy -r resource/images -w resource/cv_lists/cv_list1.txt 


