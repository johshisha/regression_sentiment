#


python program/extract_fc7.py -g 1 -init resource/alex_fc7.model -w -r resource/images resource/cv_lists/cv_list1.txt &
 
python program/extract_fc7.py -g 0 -init resource/alex_fc7.model -w -r resource/images resource/cv_lists/cv_list2.txt &

wait;

python program/extract_fc7.py -g 1 -init resource/alex_fc7.model -w -r resource/images resource/cv_lists/cv_list3.txt &

python program/extract_fc7.py -g 0 -init resource/alex_fc7.model -w -r resource/images resource/cv_lists/cv_list4.txt &

python program/extract_fc7.py -g 1 -init resource/alex_fc7.model -w -r resource/images resource/cv_lists/cv_list5.txt &

wait;

#sh ./program/cv_svc.sh &

#sh ./program/cv_svr.sh &

wait;
