# simple_baseline

基于 `baseline_itemcf_lgb.py` 思路扩展的简版 OTTO 两阶段方案。

## 结构

- `run_baseline.py`: CLI、`cv/submit` 主流程、结果写出
- `recall.py`: 数据读取、周级切分、三路召回矩阵与候选 union
- `features.py`: `w2v` 缓存、特征构造、`LGBMClassifier` 训练与评估

## 召回

三路召回：

- `top150 base covis`
- `top100 click2click`
- `top100 click2cart`

使用方式：

- `clicks = history + base_covis + click2click`
- `carts = history + base_covis + click2cart`
- `orders = history + base_covis + click2cart`

规则：

- 每一路先按各自 topK 截断
- 三路 union 去重后不再做二次截断
- 仅当 union 后候选数不足 `20` 时，才用 fallback 补足

## 排序

排序模型保持 3 个 `LGBMClassifier`：

- `clicks`
- `carts`
- `orders`

训练样本：

- 从训练集抽样 `--train-sample-sessions` 个 session
- 每个 session 随机截断为 `prefix + suffix`
- 用 `prefix` 做召回
- 用 `suffix` 生成标签
- 全保正样本，每个 query 最多保留 `20` 个负样本

## 特征

核心特征分 5 组：

- `session feats`
  - click/cart/order 次数与频率
  - last click/cart/order hour
  - last behavior type
- `aid feats`
  - aid click/cart/order count
  - aid click/cart/order ratio
  - aid last click/cart/order time gap
  - aid mean behavior type
- `session-aid feats`
  - 当前 session 内该 aid 的 click/cart/order count
  - 当前 session 内该 aid 的 last click/cart/order time gap
  - 当前 session 内该 aid 的 mean type 和 last type
  - `abs(session_aid_ratio - global_aid_ratio)` 差值特征
- `sim feats`
  - 对 `base_covis`、`click2click`、`click2cart` 三路矩阵
  - 分别对 `all/clicks/carts/orders` 四个 bucket 做 `mean/max/last` 聚合
  - 保留三路矩阵中的 best rank
- `w2v feats`
  - `candidate vs last aid`
  - `candidate vs last click aid`
  - `candidate vs last cart aid`
  - `candidate vs last order aid`

## 数据与切分

固定读取：

- `data/raw/train.jsonl`
- `data/raw/test.jsonl`

默认 `cv` 切分：

- `split_ts = max_ts - 7 * MS_PER_DAY`
- `split_ts` 前作为训练侧
- `split_ts` 后作为验证池

也支持：

- `--cv-split 2day`

## 运行

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai
python simple_baseline/run_baseline.py \
  --mode cv \
  --cv-split week \
  --train-sample-sessions 500000 \
  --num-threads 25
```

正式提交：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai
python simple_baseline/run_baseline.py \
  --mode both \
  --cv-split week \
  --train-sample-sessions 1000000 \
  --num-threads 25
```

## 输出

- `outputs/cv_metrics.json`
- `outputs/submission.csv`

`cv_metrics.json` 会额外包含：

- `coverage_by_target`
- `candidate_source_stats`
- `cv_split_mode`
- `avg_candidates_by_target`
- `w2v_cache_hit`
