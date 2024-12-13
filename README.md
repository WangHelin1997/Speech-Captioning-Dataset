## 1. Get noisy file ids

```bash
CACHE_DIR=/data/lmorove1/hwang258/gigadata/GigaCaps3/gigaspeech_val
OUTPUT_PATH=/data/lmorove1/hwang258/Speech-Captioning-Dataset/noisy_ids.txt
python get_noisy_ids.py --datadir $CACHE_DIR --savepath $OUTPUT_PATH
```

## 2. Audio tagging
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
Please change the path in `EAT/scripts/inference.sh`.


## 3. Add background tag

```bash
python 
```
