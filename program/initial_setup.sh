#mkdir
mkdir resource/cv_classification_model
mkdir resource/cv_lists
mkdir resource/cv_regression_model
mkdir resource/cv_svc_model
mkdir resource/cv_svr_model
mkdir resource/fc7_features
ln -s /media/dl-box/HD-LCU3/study/dataset/Flickr_dataset/images resource/images

#regression_label
cp /media/dl-box/HD-LCU3/study/csv/regression_label.txt resource/regression_label.txt
python program/make_cv_lists.py resource/regression_label.txt

#fc7 file
cp /home/dl-box/study/sentiment_degree.backup/resource/bvlc_alexnet.caffemodel resource/bvlc_alexnet.caffemodel
cp /home/dl-box/study/sentiment_degree.backup/resource/alex_mean.npy resource/alex_mean.npy

#make finetune model
python program/make_finetune_model.py -m 'classification' -i 'alex_model_for_classification' &
python program/make_finetune_model.py -m 'regression' -i 'alex_model_for_regression' &
python program/make_finetune_model.py -m 'fc7' -i 'alex_model_for_fc7' &

mkdir same_number
mkdir same_number/resource
ln -s /home/dl-box/study/sentiment_degree/program same_number/program

cd same_number

#mkdir
mkdir resource/cv_classification_model
mkdir resource/cv_lists
mkdir resource/cv_regression_model
mkdir resource/cv_svc_model
mkdir resource/cv_svr_model
mkdir resource/fc7_features
ln -s /media/dl-box/HD-LCU3/study/dataset/Flickr_dataset/images resource/images

#fc7 file
cp /home/dl-box/study/sentiment_degree.backup/resource/bvlc_alexnet.caffemodel resource/bvlc_alexnet.caffemodel
cp /home/dl-box/study/sentiment_degree.backup/resource/alex_mean.npy resource/alex_mean.npy

#make finetune model
python program/make_finetune_model.py -m 'classification' -i 'alex_model_for_classification' &
python program/make_finetune_model.py -m 'regression' -i 'alex_model_for_regression' &
python program/make_finetune_model.py -m 'fc7' -i 'alex_model_for_fc7' &

python program/same_number_emotion.py
python program/make_cv_lists.py resource/same_number_label.txt

cd ../

wait;


