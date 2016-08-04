#

for i in 1 2 3 4 5
do
    python program/compute_mean.py $i &
done;

wait;

sh ./program/cv_train_classification.sh &

sh ./program/cv_train_regression.sh &

sh ./program/cv_extract_features.sh &

wait;

#same_number

cd same_number

for i in 1 2 3 4 5
do
    python program/compute_mean.py $i &
done;

wait;

sh ./program/cv_train_classification.sh &

sh ./program/cv_train_regression.sh &

sh ./program/cv_extract_features.sh &

wait;

cd ../
