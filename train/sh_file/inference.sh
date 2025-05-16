# python inference.py --tp 2 --device cuda:3 --stage 10 --dt 1 --p_index 0 --sup single
# python inference.py --tp 2 --device cuda:3 --stage 10 --dt 2 --p_index 0 --sup single
# python inference.py --tp 2 --device cuda:3 --stage 10 --dt 3 --p_index 0 --sup single
# python inference.py --tp 2 --device cuda:3 --stage 10 --dt 4 --p_index 0 --sup single
# python -m torch.distributed.launch  --nproc_per_node=3  inference.py --tp 3 --device cuda:3 --stage 10 --dt 5 --p_index 0 --sup single

# python inference.py --tp 2 --device cuda:3 --stage 11 --dt 1 --p_index 1 --sup dual
# python inference.py --tp 2 --device cuda:3 --stage 11 --dt 2 --p_index 1 --sup dual
# python inference.py --tp 2 --device cuda:3 --stage 11 --dt 3 --p_index 1 --sup dual
# python inference.py --tp 2 --device cuda:3 --stage 11 --dt 4 --p_index 1 --sup dual
# python -m torch.distributed.launch --nproc_per_node=3  inference.py --tp 3 --device cuda:3 --stage 11 --dt 5 --p_index 1 --sup dual
# python inference.py --tp 1 --device cuda:2 --stage 4 --dt 1 --p_index 0 --sup allright
# python inference.py --tp 1 --device cuda:2 --stage 4 --dt 4 --p_index 0 --sup allwrong
# python inference.py --tp 4 --device cuda:2 --stage 4 --dt 0 --p_index 0 --sup all --ds 0
# python inference.py --tp 4 --device cuda:2 --stage 4 --dt 1 --p_index 0 --sup padding --ds 1
# python inference.py --tp 4 --device cuda:2 --stage 4 --dt 2 --p_index 0 --sup right --ds 2
# python inference.py --tp 1 --device cuda:2 --stage 4 --dt 7 --p_index 0 --sup all


python inference.py --tp 1 --device cuda:2 --stage 4 --dt 1 --p_index 0 --sup allright
python inference.py --tp 1 --device cuda:2 --stage 4 --dt 4 --p_index 0 --sup padding