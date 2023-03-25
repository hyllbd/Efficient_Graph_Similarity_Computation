
# net=gin
net=ginmp

fixnet=ginmp

epochs=6000
bs=128
lr=0.001
wb=0            # 1:use wandb, 0:not use wandb
cudaid=0
testinterval=1000

checkpoint="path_to_checkpoint" # path to checkpoint

dataset='AIDS700nef'
# dataset='LINUX'
# dataset='IMDBMulti'
# dataset='ALKANE'


# ******************************************** temp test ********************************************
echo "python -u src/main_kd.py --dataset $dataset --gnn-operator $net --gnn-operator-fix $fixnet --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval --checkpoint $checkpoint"
python -u src/main_kd.py --dataset $dataset --gnn-operator $net --gnn-operator-fix $fixnet --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval --checkpoint $checkpoint \
# > 'logs/train.log' 2>&1


# ******************************************** not nohup ********************************************
# dataset='AIDS700nef'
# python -u src/main_kd.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
# > 'logs/'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$cudaid'_'$testinterval'.log' 2>&1 &



# ******************************************** nohup ********************************************
# dataset='AIDS700nef'
# nohup python -u src/main_kd.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
# > 'logs/'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$cudaid'_'$testinterval'.log' 2>&1 &





# check process
# ps -ef | grep "python -u src/main_kd.py" | grep -v grep