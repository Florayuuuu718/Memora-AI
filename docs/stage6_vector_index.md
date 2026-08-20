# Stage 6：FAISS / HNSW / Qdrant 向量索引

## 1. 实验目标

保留 NumPy 暴力搜索作为 exact baseline，再比较本地索引和向量数据库后端：

```text
NumPy Brute Force
FAISS Flat
FAISS HNSW
Qdrant HNSW
```

主要指标：Recall@10、P95 Latency 和 Memory。

## 2. 实验设置

| 项目 | 设置 |
|---|---:|
| 图片数量 | 126 |
| OpenCLIP 向量维度 | 512 |
| 查询数量 | 100 |
| Recall 基准 | NumPy Exact |
| Qdrant 模式 | 内存模式 |

## 3. 实验结果

| 后端 | Recall@10 | P95 Latency | Memory |
|---|---:|---:|---:|
| NumPy Exact | 1.000 | 0.0470 ms | 0.2461 MB |
| FAISS Flat | 1.000 | **0.0188 ms** | 0.2461 MB |
| FAISS HNSW | 0.998 | 0.0589 ms | 0.2538 MB |
| Qdrant HNSW | 1.000 | 0.4380 ms | 0.2538 MB |

## 4. 如何解读

- NumPy Exact 是召回率基准，当前 Recall@10 为 1.000；
- FAISS Flat 在这 126 张照片上最快，且保持 1.000 召回；
- FAISS HNSW 出现极小的召回下降（0.998），但当前数据规模还不能体现 HNSW 的规模优势；
- Qdrant HNSW 可以运行并保持 1.000 召回，但本次是内存模式，测得的 Memory 是索引估算，不是独立 Qdrant 服务的真实 RSS；
- 不能根据这次小数据集结果断言生产环境一定选择 FAISS Flat。需要扩大到至少数千或上万张照片后重新测 P95、内存和构建时间。

## 5. 运行方式

```powershell
python scripts/benchmark.py `
  --index-path .\data\index-openclip-prepared.json `
  --queries 100
```

如果要测试增强索引，将路径替换为 `data/index-openclip-prepared-v4-enriched.json`。NumPy exact baseline 需要保留，不能因为接入 FAISS 或 Qdrant 而删除。

## 6. 当前结论

Stage 6 已完成小规模可运行 benchmark：四种后端均可用，FAISS Flat 当前最快，NumPy Exact 继续作为准确率基准。下一步是扩大数据规模并测量真实服务部署下的内存和 P95 延迟，再决定是否默认使用 FAISS、HNSW 或 Qdrant。
