#for N in {1..2}; do \
python3 src/align/align_dataset_mtcnn.py \
../facenet/datasets/barry/raw \
../facenet/datasets/barry/align_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
#& done
