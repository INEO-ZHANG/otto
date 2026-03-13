# kaggle otto

## baseline

```
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai
python baseline_itemcf_lgb.py \
  --data-zip ./otto-recommender-system.zip \
  --mode both \
  --train-sample-sessions 500000 \
  --num-threads 32
```

输出目录：

- `data/raw/`: 解压后的官方原始文件
- `outputs/cv_metrics.json`: 本地验证指标
- `outputs/submission.csv`: Kaggle 提交文件
