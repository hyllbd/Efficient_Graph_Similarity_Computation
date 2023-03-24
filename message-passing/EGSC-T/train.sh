
# net=gin
# net=mpnn
net=ginmp
# net=ginskip

epochs=10
bs=128
lr=0.001
wb=1            # 1:use wandb, 0:not use wandb
cudaid=0
testinterval=10

# ******************************************** temp test ********************************************
dataset='AIDS700nef'
python -u src/main.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
> 'logs/train.log' 2>&1

# dataset='LINUX'
# python -u src/main.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
# # > 'logs/train.log' 2>&1

# dataset='IMDBMulti'
# python -u src/main.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
# > 'logs/train.log' 2>&1

# dataset='ALKANE'
# python -u src/main.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
# > 'logs/train.log' 2>&1


# ******************************************** not nohup ********************************************
# dataset='AIDS700nef'
# python -u src/main.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
# > 'logs/'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$cudaid'_'$testinterval'.log' 2>&1 &


# dataset='LINUX'
# python src/main.py  --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --wandb $wb --plot > 'logs/log_'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$i'.txt'


# dataset='IMDBMulti'
# python src/main.py  --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --wandb $wb --plot > 'logs/log_'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$i'.txt'


# dataset='ALKANE'
# python src/main.py  --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --wandb $wb --plot > 'logs/log_'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$i'.txt'



# ******************************************** nohup ********************************************
dataset='AIDS700nef'
nohup python -u src/main.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval \
> 'logs/'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$cudaid'_'$testinterval'.log' 2>&1 &














# check process
# ps -ef | grep "python -u src/main.py" | grep -v grep