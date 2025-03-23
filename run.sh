torchrun \
--nnodes 1 \
--nproc_per_node 3 \
./dist_train.py

# torchrun \
# --nnodes 1 \
# --nproc_per_node 3 \
# --rdzv_id $RANDOM \
# --rdzv_backend c10d \
# --rdzv_endpoint $head_node_ip:29500 \
# ./dist_train.py