# Redis Query Embedding Cache

## 为什么引入 Redis

在 AI 应用里，很多查询会重复触发同一段 query embedding 计算。  
Redis 适合做这类短生命周期缓存，可以减少重复调用 embedding 接口，降低延迟和成本。

## 本项目当前用途

当前只缓存**用户 query embedding**，不缓存文档入库时的 chunk embedding，不缓存最终 LLM 回答。

缓存 key 设计：

`embedding:{embedding_model}:{sha256(query)}`

示例：

`embedding:text-embedding-3-small:9a1f...`

## 为什么 Redis 不替代 PostgreSQL / pgvector

- PostgreSQL / pgvector 是文档与向量检索的主存储，负责持久化与检索排序。
- Redis 在这里是易失性加速层，只做短期命中优化。
- 即使 Redis 清空或不可用，主流程仍由现有数据库能力保证。

## Redis 不可用时如何降级

`agent/cache_utils.py` 中所有 Redis 读写都做了异常捕获：

- 连接失败/读写失败时返回 `None` 或 `False`
- 调用方继续走原 embedding 逻辑
- 不会因为 Redis 故障导致问答或入库失败

## 环境变量

- `REDIS_URL`，默认 `redis://redis:6379/0`
- `ENABLE_REDIS_CACHE`，默认 `true`
- `EMBEDDING_CACHE_TTL_SECONDS`，默认 `86400`

## 后续可扩展方向

- 检索结果缓存（按 query + 检索参数）
- 轻量限流（按用户/IP/session 的窗口计数）
- 任务状态读缓存（降低高频轮询数据库压力）

> 当前实现保持最小改动，优先满足“可降级、低风险、易验证”。
