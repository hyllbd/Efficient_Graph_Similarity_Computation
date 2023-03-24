# linux 后台执行多条命令，不打印输出到终端

cd ~/code/Efficient_Graph_Similarity_Computation/EGSC-KD/

# # epochs=3000
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=3000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 

# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=3000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 

# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=3000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=16 

# epochs=6000
echo "nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=0 --light=0"
nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=0 --light=0

echo "nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 --light=0"
nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 --light=0

echo "nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 --light=1"
nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 --light=1

echo "nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=16 --cuda-id=2 --light=0"
nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=16 --cuda-id=2 --light=0


# epochs=7000
echo "nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=7000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=3 --light=0"
nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=7000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=3 --light=0


# # epochs=8000
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=8000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 

# # epochs=9000
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=9000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=2 

# # epochs=10000
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=10000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=3 