# 推荐系统召回阶段中的 C++ 服务端技术

## 1. 服务端视角下的召回系统

召回阶段本质上是一个低延迟、高并发、多路并行查询、大规模数据检索、强容错且可热更新的在线服务系统。

```text
推荐请求
        ↓
召回编排服务
   ├─ 双塔 ANN 召回
   ├─ ItemCF / Swing
   ├─ 作者召回
   ├─ 地理召回
   ├─ 热门 / 缓存召回
   └─ 特征与过滤服务
        ↓
合并、去重、过滤
        ↓
粗排
```

---

## 2. RPC 与 Protobuf

不同召回通道通常是独立服务，通过 RPC 调用。

常见技术：

```text
gRPC、brpc、Thrift、Protobuf、自研 RPC
```

主要用于调用特征服务、ANN 服务、ItemCF 服务、作者召回、地理召回等。重点关注连接复用、Deadline、超时、重试、序列化成本和请求大小。

---

## 3. 异步 I/O 与多路并行召回

多路召回必须并行，而不是串行。并行后，总延迟接近最慢一路，而不是所有延迟之和。

C++ 常见实现：

```text
线程池、Future / Promise、C++20 协程、异步 RPC、epoll / 事件循环
```

核心原则：等待网络和 KV 时，不要长期占用工作线程。

---

## 4. 线程池与资源隔离

常见拆分：

```text
I/O 线程：RPC、Redis、远程 KV
计算线程池：ANN、TopK、去重、过滤
后台线程：索引更新、配置刷新、指标上报
```

需要避免慢请求占满线程池、CPU 计算阻塞 I/O 线程、后台任务影响在线请求。常见措施包括有界队列、并发上限、线程池隔离、优先级调度和 CPU 绑核。

---

## 5. KV 与缓存

召回阶段常读取：

```text
用户 Embedding、用户画像、最近行为、ItemCF 相似列表、作者关系、物品状态、热门列表
```

常见存储：

```text
Redis、RocksDB、LevelDB、分布式 KV、本地内存缓存
```

典型映射：

```text
user_id -> user_embedding
user_id -> recent_items
item_id -> similar_items
author_id -> latest_items
geo_cell -> nearby_items
```

优化重点包括本地缓存 + 远程 KV、批量 MGet、热点 Key 副本、请求合并、TTL 随机抖动和异步刷新。

---

## 6. ANN 索引服务

双塔召回通常使用 Faiss、hnswlib、HNSW 或 IVF-PQ。

服务端重点包括：

```text
索引加载、索引分片、多副本、线程安全、批量查询、SIMD、增量更新
```

如果索引过大，可以按 item 分片，并行查询各分片局部 TopK，再合并成全局 TopK。

---

## 7. 内存管理与数据布局

召回服务处理大量向量和候选数据，因此非常依赖内存效率。

重点技术：

```text
连续内存、reserve 预分配、对象池、内存池、Arena、减少 new/delete、避免不必要拷贝
```

向量通常更适合连续存储，以提升 CPU Cache 命中率，并便于 SIMD 和批量计算。

---

## 8. SIMD 与数值计算

向量召回会大量执行点积、余弦相似度、欧氏距离和 TopK。

常见优化：

```text
AVX2、AVX-512、NEON、编译器自动向量化、float16 / int8 量化、内存对齐、Prefetch
```

实际工程中通常优先使用 Faiss、hnswlib 等成熟库的优化实现。

---

## 9. 候选合并、去重与过滤

多个通道会召回相同物品，需要合并来源。

```cpp
struct Candidate {
    uint64_t item_id;
    float vector_score;
    float itemcf_score;
    uint32_t source_mask;
};
```

常用结构：

```text
unordered_map、flat_hash_map、Bitmap、Bloom Filter、小根堆 TopK、排序归并
```

过滤内容包括已看、已买、下架、库存不足、黑名单、地域限制和低质量内容。

---

## 10. Bloom Filter

Bloom Filter 常用于快速过滤用户已经看过的物品。

特点：

```text
内存占用小
查询快
不会漏掉真正已存在的元素
可能误判少量未看物品
```

适合做第一层快速过滤，必要时可以再查询精确历史集合。

---

## 11. 批量查询与 Micro-Batching

不要对 100 个 item 发 100 次 RPC，应使用 BatchGet 一次查询。批量化可以减少网络往返、系统调用、序列化次数和锁竞争。

ANN 也可以进行 Micro-Batching，但需要平衡吞吐、延迟、Batch 大小和最大等待时间。

---

## 12. 分片、路由与负载均衡

常见分片方式：

```text
user_id 分片、item_id 分片、场景分片、地域分片
```

需要处理一致性哈希、扩缩容、数据迁移、热点分片和多副本读取。ANN 通常按 item 分片，用户 KV 通常按 user_id 分片。

---

## 13. 超时、熔断、降级与兜底

某一路召回失败不应拖垮整个请求。

```text
双塔超时
        ↓
仍然使用 ItemCF、热门和缓存结果
```

常见机制：

```text
每路独立超时、整体 Deadline、熔断、限流、降级、兜底
```

召回服务更关注尽快返回可用结果，而不是很晚返回完美结果。

---

## 14. 限流与背压

高峰期请求超过处理能力时，必须限制进入系统的任务量。

常见技术：

```text
令牌桶、漏桶、有界任务队列、并发上限、快速失败、背压
```

否则会出现队列膨胀、延迟增加、内存上涨和服务雪崩。

---

## 15. 模型与索引热更新

ANN 索引和模型需要不断更新，不能停机替换。

常见双缓冲方案：

```text
Index A 在线服务
        ↓
后台加载 Index B
        ↓
校验与预热
        ↓
原子切换指针
        ↓
新请求使用 Index B
        ↓
旧请求结束后释放 Index A
```

C++ 常用 `shared_ptr`、原子指针和 RCU 思想。必须保证用户塔、物品塔、item embedding 和 ANN 索引版本兼容。

---

## 16. 配置中心与动态参数

召回参数通常需要在线调整：

```text
每路 TopK、超时时间、ANN efSearch、过滤规则、热门兜底数量、多路召回配额
```

常见配置中心包括 etcd、ZooKeeper、Apollo、Nacos 和自研配置中心，用于动态调参、灰度发布和按场景配置。

---

## 17. 可观测性

重点指标：

```text
QPS、平均延迟、P95 / P99、超时率、错误率、每路召回数量、去重率、过滤率、缓存命中率、ANN 延迟、线程池队列长度、CPU / 内存
```

常见工具：

```text
结构化日志、分布式 Trace、Prometheus、Grafana、OpenTelemetry
```

---

## 18. 核心技术总结

### 网络与并发

```text
RPC、Protobuf、epoll、线程池、协程、异步 I/O
```

### 数据与缓存

```text
Redis、RocksDB、分布式 KV、本地缓存、Bloom Filter、批量查询
```

### 高性能计算

```text
ANN、SIMD、连续内存、内存池、TopK、Hash 去重
```

### 服务治理

```text
超时、限流、熔断、降级、分片、负载均衡
```

### 工程与运维

```text
索引热更新、版本管理、配置中心、灰度发布、监控与链路追踪
```

从服务端角度看，召回系统的核心目标是：

```text
快：异步并行、缓存、SIMD、批处理
稳：超时、熔断、限流、降级
大：分片、分布式 KV、ANN 多副本
可更新：模型和索引热切换
```
