
net=gin
# net=mpnn
# net=ginmp
# net=ginskip

fixnet=ginmp

epochs=6000
bs=128
lr=0.001
wb=0            # 1:use wandb, 0:not use wandb
cudaid=0
testinterval=100

dataset='AIDS700nef'
checkpoint="../Checkpoints/AIDS700nef/EGSC_g_EarlyFusion_AIDS700nef_ginmp_1.37061_20000_128_0.001_checkpoint.pth"

# dataset='LINUX'
# checkpoint="../Checkpoints/LINUX/EGSC_g_EarlyFusion_LINUX_ginmp_0.23924_6000_128_0.001_checkpoint.pth"

# dataset='IMDBMulti'
# checkpoint="../Checkpoints/IMDBMulti/EGSC_g_EarlyFusion_IMDBMulti_ginmp_8.22481_10_128_0.001_checkpoint.pth"

# dataset='ALKANE'
# checkpoint="../Checkpoints/ALKANE/EGSC_g_EarlyFusion_ALKANE_ginmp_0.86028_10000_128_0.001_checkpoint.pth"


# ******************************************** temp test ********************************************
python -u src/main_kd.py --dataset $dataset --gnn-operator $net --gnn-operator-fix $fixnet --epochs $epochs --batch-size $bs --learning-rate $lr --plot --wandb $wb --cuda-id=$cudaid --test-interval $testinterval --checkpoint $checkpoint \
> 'logs/train.log' 2>&1


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