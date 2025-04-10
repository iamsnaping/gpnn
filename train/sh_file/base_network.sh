#python train_pretrain_all.py --stage 1 --prompt 1 --lr 2e-4 --sup base_2 --device cuda:2 --ds 2
python train_pretrain_all.py --stage 3 --sup basenetwork_graph --lr 2e-4  --device cuda:2 --epoch 20 --batchsize 64
# python train_pretrain_all.py --stage 5 --sup basenetwork_no_graph --lr 2e-4  --device cuda:2 --epoch 5 --batchsize 64