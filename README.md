## 1. Drop columns for fast processing
```bash
python clean.py
```

## 2. Get noisy file ids

```bash
python get_noisy_ids.py
```

## 3. Audio tagging
Install
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
git clone https://github.com/WangHelin1997/EAT
```
Download [model](https://drive.google.com/file/d/1b_f_nQAdjM1B6u72OFUtFiUu-4yM2shd/view?usp=sharing)

```bash
bash EAT/scripts/inference.sh 
```


## 4. Add background tag

```bash
python add_background_tags.py
```

