CUDA_VISIBLE_DEVICES=1 \
python retrieve.py \
    --batch_size 1 \
    --epoch_nums 5 \
    --learning_rate 2e-6 \
    --is_train 1 \
    --is_test 0 \
    --is_firststage 0 \
    --output_dir retrieve2 \
    --load_dir retrieve1 