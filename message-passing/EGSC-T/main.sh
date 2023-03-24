# linux 后台执行多条命令，不打印输出到终端

cd /home/zl525/code/Efficient_Graph_Similarity_Computation/EGSC-T && conda activate Efficient-Graph-Similarity-Computation

# 测试用
python src/main.py --dataset=AIDS700nef --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1
python src/main.py --dataset=LINUX --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1
python src/main.py --dataset=IMDBMulti --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1
python src/main.py --dataset=ALKANE --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1


############################################AIDS700nef epochs=6000 ############################################
# 正在跑
nohup python src/main.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &

############################################LINUX epochs=6000 ############################################
# 正在跑
nohup python src/main.py --dataset=LINUX --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &


############################################IMDBMulti epochs=6000 ############################################
# 正在跑
nohup python src/main.py --dataset=IMDBMulti --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &


############################################ALKANE epochs=6000 ############################################
# 正在跑
nohup python src/main.py --dataset=ALKANE --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &
