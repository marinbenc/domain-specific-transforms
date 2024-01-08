python -m seg.train_seg --subset dermquest --batch-size 16 --lr 0.0001 --epochs 100 --log-name dermquest_final &&
python -m fine_tune --subset dermquest --batch-size 16 --lr 0.0001 --epochs 100 --log-name dermquest_final &&
python -m seg.train_seg --subset dermis --batch-size 16 --lr 0.0001 --epochs 100 --log-name dermquest_final &&
python -m fine_tune --subset dermis --batch-size 16 --lr 0.0001 --epochs 100 --log-name dermis_final &&
python -m fine_tune --subset isic --batch-size 16 --lr 0.0001 --epochs 100 --log-name isic_final &&
python -m fine_tune --subset isic --batch-size 16 --lr 0.0001 --epochs 100 --log-name isic_final
