

for i in 2 3
do
python program/train_regression_normal_distribution.py -g 1 --no-show -r 'resource/images' --initmodel resource/alex_regression.model resource/cv_lists $i &

done;

wait;
for i in 2 3
do

python program/estimate_regression.py -g 1 -init resource/cv_regression_normal_distribution_model/remove_${i}/model -m resource/cv_lists/remove_${i}_mean.npy -r resource/images -w resource/cv_lists/cv_list${i}.txt

python program/estimate_regression.py -g 1 -init resource/cv_regression_model/remove_${i}/model -m resource/cv_lists/remove_${i}_mean.npy -r resource/images -w resource/cv_lists/cv_list${i}.txt

done;


for i in 4 5
do
python program/train_regression_normal_distribution.py -g 0 --no-show -r 'resource/images' --initmodel resource/alex_regression.model resource/cv_lists $i &

python program/train_regression.py -g 1 --no-show -r 'resource/images' --initmodel resource/alex_regression.model resource/cv_lists $i &
wait;

done;

for i in 4 5
do

python program/estimate_regression.py -g 1 -init resource/cv_regression_normal_distribution_model/remove_${i}/model -m resource/cv_lists/remove_${i}_mean.npy -r resource/images -w resource/cv_lists/cv_list${i}.txt

python program/estimate_regression.py -g 1 -init resource/cv_regression_model/remove_${i}/model -m resource/cv_lists/remove_${i}_mean.npy -r resource/images -w resource/cv_lists/cv_list${i}.txt

done;
wait;


python program/calc_performance_from_log.py resource/cv_regression_normal_distribution_model/remove_1/result_model.txt

python program/calc_performance_from_log.py resource/cv_regression_model/remove_1/result_model.txt

