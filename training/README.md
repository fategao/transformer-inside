# Training notes

`train.py` trains the 4-layer character-level Transformer used by the site and
exports the browser artifacts into `public/model/`.

```bash
python training/train.py
python training/train.py --steps 3000 --device cpu
```

The corpus is downloaded once into `training/data/tinyshakespeare.txt` (ignored
by git). The exported weights are not ignored: they are the content the site
serves. `training/checkpoint.pt` is a local convenience checkpoint and should
not be committed.
