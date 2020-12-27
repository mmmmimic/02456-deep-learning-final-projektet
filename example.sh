# ENZYMES
# python -m train --datadir=data --bmname=ENZYMES --cuda=0 --max-nodes=100 --num-classes=6

# ENZYMES - Diffpool
#python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=0 --epochs=1 --num-classes=6 --method=soft-assign --dropout=0.8

# DD
# python -m train --datadir=data --bmname=DD --cuda=0 --max-nodes=500 --epochs=1000 --num-classes=2

# DD - Diffpool
#python -m train --bmname=DD --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --cuda=0 --num-classes=2 --method=soft-assign --epochs=1

!python -m train --bmname=DD --batch-size=30 \
--dropout=0.3 --assign-ratio=0.5 --unpool-ratio=0.5 \
--hidden-dim=64 --output-dim=64 --cuda=0 --num-classes=2 \
--method=soft-assign --num-pool=3 --epochs=100 --num-unpool=3 \
--weight-decay=0