# GDA-Project
Charles Monté &amp; Elouan Gardès, 
- The Details Matter: Preventing Class Collapse in Supervised Contrastive Learning [paper link](https://mdpi-res.com/d_attachment/csmf/csmf-03-00004/article_deploy/csmf-03-00004.pdf?version=1650444797)
- [Our paper](https://www.overleaf.com/4548983824dskwwxgsthcy#58a21c)

Load data:
- run "wget -O waterbirds.tar.gz [copy this link](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)"
- then extract with "tar -xvzf waterbirds.tar.gz"

Place this data in the parent folder and the code should be usable as is. No requirements is added as any torch, torchvision packages should work and no additionnal exotic library is used.

- reproduce is a notebook where experiments can be run to reproduce the original paper's experiments.
- experiments_waterbirds is a python file aimed at running experiments all at once and display results/create plots.
- cifar100[] is a notebook aimed at reproducing experiments from the original paper. It fails at doing so.
- utils contain plot function, datasets loading and other utilities used in the rest of this repo.
