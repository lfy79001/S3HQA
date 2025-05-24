#CUDA_VISIBLE_DEVICES=8 \
python retrieve.py \
    --batch_size 1 \
    --epoch_nums 5 \
    --learning_rate 7e-6 \
    --is_train 1 \
    --rerank_link 0 \
    --is_test 0 \
    --is_firststage 1 \
    --output_dir retrieve1 \
    --load_dir retrieve1 