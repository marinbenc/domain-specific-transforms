#python -m seg.train_seg --subset isic --batch-size 48 --lr 0.0001 --epochs 100 --log-name isic_aug_better && \
#python -m stn.train_stn --lr 0.00001 --batch-size 64 --epochs 200 --subset isic --log-name isic_aug_better && \
#python -m fine_tune --subset isic --batch-size 48 --lr 0.00000001 --epochs 100 --log-name isic_aug_better

python -m seg.train_seg --subset isic --batch-size 16 --lr 0.0001 --epochs 100 --log-name isic_512 && \
python -m stn.train_stn --lr 0.00001 --batch-size 16 --epochs 200 --subset isic --log-name isic_512 && \
python -m fine_tune --subset isic --batch-size 8 --lr 0.00000001 --epochs 100 --log-name isic_512