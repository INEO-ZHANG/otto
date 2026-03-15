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
可以把当前方案理解成一个“多路 itemCF 召回 + 手工特征/embedding 特征重排 + 双模型融合”的 OTTO 排序系统。

**方案结构**
入口在 [run_baseline.py](/root/autodl-tmp/otto/src/run_baseline.py#L130)。整体流程是：

1. 读 `train.jsonl` / `test.jsonl`
2. `cv` 时按最近 2 天做离线切分，`submit` 时用全量训练集
3. 基于全量可用训练语料构建多矩阵 covis 和热门度统计
4. 基于同一份语料生成 embedding cache
5. 从训练 session 抽样 `train-sample-sessions` 个 session，随机截断成 `prefix + future labels`
6. 用 `prefix` 做召回，生成候选
7. 对候选算手工特征 + embedding 特征
8. 训练两套排序模型：
   - `LGBMClassifier`
   - `CatBoostRanker(GPU)`
9. 推理时按 query 做 `RRF` 融合，输出 top20

**召回层**
召回核心在 [itemcf.py](/root/autodl-tmp/otto/src/itemcf.py#L242) 和 [itemcf.py](/root/autodl-tmp/otto/src/itemcf.py#L480)。

当前不是单一 itemCF，而是 3 张语义矩阵：
- `cart_order`
- `click_buy`
- `click_cart_order`

每条 item-item 边的权重同时考虑：
- 交互类型组合 `pair_weights`
- 共现时间差 `time_scale_ms`
- 顺序方向 `backward_scale`
- 序列位置距离 `1 / (1 + gap)`
- 热门度去偏置 `pop_alpha`

这比普通“相邻共现计数”强很多，因为它显式建模了：
- click 到 buy 的转化关系
- cart/order 的高意图关系
- 时间近邻和方向性

目标融合权重在 [itemcf.py](/root/autodl-tmp/otto/src/itemcf.py#L105)：
- `clicks` 更依赖 `click_cart_order`
- `orders` 更依赖 `cart_order`
- `carts` 当前偏向 buy 信号：`0.60 / 0.30 / 0.10`

召回时：
- 先保留 session 最近 `20` 个去重 aid
- 再从最近行为扩 covis 邻居
- `carts` 目前单独增强了召回：
  - 用最近 `10` 个 source item 扩邻居
  - 候选上限 `100`
- 其它目标还是最近 `5` 个 source、候选上限 `60`

这个调整在 [itemcf.py](/root/autodl-tmp/otto/src/itemcf.py#L487)。

**特征层**
特征定义在 [features.py](/root/autodl-tmp/otto/src/features.py#L28)。

大体分 4 类：

1. `itemCF` 召回强度特征
- `itemcf_score_sum`
- `itemcf_score_max`
- `itemcf_best_rank`

2. session-candidate 交互特征
- 候选是否在历史中出现过
- `click/cart/order` 分别出现多少次
- 是否是最后一个 aid
- 最近/最早出现位置
- 距上次出现多久

3. session 上下文特征
- `session_len`
- `session_unique_aids`
- `session_click_count`
- `session_cart_count`
- `session_order_count`
- `last_event_type`
- `time_since_last_event`

4. 热门度与局部协同特征
- `global_pop_rank_all`
- `global_pop_rank_target`
- `last_item_itemcf_score`
- `last_hour_avg_itemcf_score`
- `last_item_cocount`

**embedding 特征**
embedding 逻辑在 [embeddings.py](/root/autodl-tmp/otto/src/embeddings.py#L22)。

现在有两类：

1. `Word2Vec` item embedding
- 用 session aid 序列训练
- 只压缩连续重复 aid，保留行为顺序
- 用于 `候选商品 vs 最后一个商品` 的余弦相似度
- 特征名：`w2v_last_aid_cosine`

2. `ProNE-style` 图 embedding
- 输入是基于 `pair_counts` 的 aid-aid 稀疏图
- 实现是轻量化的“归一化图 + SVD + 图传播增强”
- 严格说更适合在简历里写成 `ProNE-style graph embedding`，比直接写“官方 ProNE”更稳妥
- session 向量定义为最近 `10` 个去重 aid 的加权平均图向量
- 特征名：`prone_session_aid_cosine`

embedding 都有缓存：
- `cv` 用 `split_ts` 前语料
- `submit` 用全量语料
- 缓存在 `cache/embeddings/`

**排序模型**
训练逻辑在 [features.py](/root/autodl-tmp/otto/src/features.py#L399)。

当前是双模型：

- `LGBMClassifier`
  - 学 pointwise 二分类
  - 优点是稳定、快、好调

- `CatBoostRanker(GPU)`
  - `task_type='GPU'`
  - `loss_function='PairLogit'`
  - query 是 `session_target`
  - 用 group 训练 query 内排序

训练样本不是全量 session，而是：
- 从训练集抽样 `train-sample-sessions`
- 每个 session 随机截断
- 基于召回候选生成训练行
- 全保正样本
- 每个 query 最多保留 `20` 个负样本

这部分在 [features.py](/root/autodl-tmp/otto/src/features.py#L281)。

**融合与推理**
推理不是直接平均分数，而是 `RRF` 融合，在 [features.py](/root/autodl-tmp/otto/src/features.py#L495)。

原因是两套模型分数尺度不同：
- `LGBMClassifier` 输出接近概率
- `CatBoostRanker` 输出原始 ranking score

所以按 rank 融合更稳：
- `LGBM weight = 0.6`
- `CatBoost weight = 0.4`

推理层还有两点工程优化：
- batch feature build
- batch predict

LGBM 侧还支持 `tl2cgen` 编译推理，在 [features.py](/root/autodl-tmp/otto/src/features.py#L366)。

**离线验证设计**
`cv` 不是随便切，而是 OTTO 常用的近似官方切法，在 [itemcf.py](/root/autodl-tmp/otto/src/itemcf.py#L371)：
- `split_ts = max_ts - 2 days`
- 用 `split_ts` 前的数据建训练语料
- 后面的 session 做伪验证
- 再随机截断出 prefix/label

这点写在简历里很有价值，因为说明你做的是“召回+排序”的离线近似评测，不是只训练一个分类器。

**如果写在简历里，建议这样表述**
可以写成 2-3 条高密度版本：

- 设计并实现 OTTO 推荐系统 baseline：基于 `click_buy / cart_order / click_cart_order` 三语义 covis 矩阵做候选召回，结合时间衰减、方向权重、位置衰减和热门度去偏置，支持 target-aware 融合召回。
- 构建多层排序特征体系，包括 itemCF 强度、session 行为上下文、热门度、局部共现，以及 `Word2Vec` 商品相似度和 `ProNE-style` 图 embedding 的 session-item 相似度特征。
- 搭建 `LGBMClassifier + GPU CatBoostRanker` 双模型重排与 `RRF` 融合方案，支持 batch 推理、`tl2cgen` 加速和 embedding cache，形成可跑 `cv/submit` 的端到端推荐流水线。

如果你想更“简历导向”，我下一条可以直接帮你压成：
1. 一版中文简历表述
2. 一版英文简历表述
3. 一版面试时的 1 分钟项目介绍