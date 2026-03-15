# kaggle otto

当前 `src` 版本是：

- 三语义 covis 召回：`cart_order`、`click_buy`、`click_cart_order`
- 排序特征：基础手工特征 + `Word2Vec` / `ProNE` embedding 相似度
- 排序模型：`LGBMClassifier` + `CatBoostRanker(GPU)` 融合
- 推理：`LightGBM` 可选 `tl2cgen` 加速；`CatBoost` 走原生预测

代码固定读取：

- `data/raw/train.jsonl`
- `data/raw/test.jsonl`

## 依赖

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai
conda install -y gensim
```

当前环境需要：

- `lightgbm`
- `catboost`
- `gensim`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `tqdm`
- 可选：`treelite` + `tl2cgen`

## 运行

推荐先跑 `cv`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai
python src/run_baseline.py \
  --mode cv \
  --matrix-mode target_aware \
  --experiment-name embed_cb_v1 \
  --train-sample-sessions 500000 \
  --num-threads 25
```

正式提交流程：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai
python src/run_baseline.py \
  --mode both \
  --matrix-mode target_aware \
  --experiment-name embed_cb_submit_v1 \
  --train-sample-sessions 500000 \
  --num-threads 25
```

如果只想加速 `LightGBM` 推理分支：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai
python src/run_baseline.py \
  --mode cv \
  --matrix-mode target_aware \
  --predict-backend tl2cgen \
  --predict-batch-size 1024 \
  --experiment-name embed_cb_tl2cgen_v1 \
  --train-sample-sessions 500000 \
  --num-threads 25
```

## 说明

- `CatBoostRanker` 固定使用 `GPU:0`
- `LGBMClassifier` 与 `CatBoostRanker` 采用按 query 的加权 `RRF` 融合
- `Word2Vec` 和 `ProNE` 都会缓存到 `cache/embeddings/`
- `cv` 使用 `split_ts` 之前的训练语料生成 embedding；`submit` 使用完整训练集
- `train-sample-sessions` 只控制排序模型训练 session 抽样，不影响 covis 和 embedding 的全量统计

当前新增 embedding 特征：

- `w2v_last_aid_cosine`
- `prone_session_aid_cosine`

## 常用参数

- `--mode {cv,submit,both}`: 本地验证、正式提交，或两者都跑
- `--matrix-mode {target_aware,one_hot}`: covis 权重方案
- `--experiment-name`: 输出文件标签
- `--train-sample-sessions`: 排序模型训练 session 抽样数
- `--num-threads`: `Word2Vec`、`LightGBM`、`tl2cgen` 使用的线程数
- `--predict-backend {lightgbm,tl2cgen}`: 仅切换 `LGBM` 推理后端
- `--predict-batch-size`: batch 特征构造与 batch 预测的 session 数

## 输出

- `outputs/cv_metrics_<experiment>.json`
- `outputs/submission_<experiment>.csv`

`cv_metrics` 里额外会写：

- `embedding_cache_hit`
- `catboost_device`
- `fusion_method`

旧版单文件参考实现保留在 `baseline_itemcf_lgb.py`。
