#

python program/train_regression_divided_case.py -g 1 --no-show resource/cv_lists 1

python program/estimate_regression.py -g 1 -init resource/cv_regression_divided_case_model/remove_1/model -m resource/cv_lists/remove_1_mean.npy -r resource/images -w resource/cv_lists/cv_list1.txt

#python program/calc_performance_from_log.py resource/cv_regression_divided_case_model/remove_1/result_model.txt --graph

<<COMMENTOUT
#python program/train_regression_divided_case.py -g 1 --no-show resource/cv_lists 1 &

python program/train_regression_divided_case.py -g 1 --no-show resource/cv_lists 2 &

python program/train_regression_divided_case.py -g 1 --no-show resource/cv_lists 3 &

wait;

python program/train_regression_divided_case.py -g 1 --no-show resource/cv_lists 4 &

python program/train_regression_divided_case.py -g 1 --no-show resource/cv_lists 5 &
wait;

#python program/estimate_regression.py -g 1 -init resource/cv_regression_divided_case_model/remove_1/model -m resource/cv_lists/remove_1_mean.npy -r resource/images -w resource/cv_lists/cv_list1.txt &

wait;

python program/estimate_regression.py -g 1 -init resource/cv_regression_divided_case_model/remove_2/model -m resource/cv_lists/remove_2_mean.npy -r resource/images -w resource/cv_lists/cv_list2.txt &

python program/estimate_regression.py -g 1 -init resource/cv_regression_divided_case_model/remove_3/model -m resource/cv_lists/remove_3_mean.npy -r resource/images -w resource/cv_lists/cv_list3.txt &

wait;

python program/estimate_regression.py -g 1 -init resource/cv_regression_divided_case_modell/remove_4/model -m resource/cv_lists/remove_4_mean.npy -r resource/images -w resource/cv_lists/cv_list4.txt &

python program/estimate_regression.py -g 1 -init resource/cv_regression_divided_case_model/remove_5/model -m resource/cv_lists/remove_5_mean.npy -r resource/images -w resource/cv_lists/cv_list5.txt &

wait;
COMMENTOUT