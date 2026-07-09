# Engineering Showcase Eval

这份报告用于展示项目的工程化链路：来源展示、请求中间件、缓存降级、多轮记忆和可观测性。

## Summary

- public_status: SHOWCASE_READY
- pass_count: 7 / 7
- total_passed_tests: 54
- runtime_seconds: 15.409

## Scorecard

| Suite | Status | Showcase | Passed Tests | Responsibility |
|---|---|---|---:|---|
| source_display_and_citation | PASS | Evidence references + citation review | 3 | 证据来源是否能生成可展示引用，并检查非法/缺失 citation |
| http_middleware_runtime | PASS | HTTP middleware + runtime metrics | 9 | 请求 ID、安全头、限流、请求大小限制和结构化错误是否可用 |
| redis_cache_degrade | PASS | Redis cache hit/miss/degrade | 7 | Redis 不可用时缓存层是否安全降级，不阻塞主流程 |
| session_memory | PASS | Multi-turn memory/runtime policy | 8 | 多轮会话摘要、历史过滤和最近消息预览是否稳定 |
| chat_metrics | PASS | Chat observability metrics | 2 | 聊天请求指标、来源混合、工具调用和最近请求统计是否稳定 |
| runtime_config_and_providers | PASS | Runtime diagnostics + provider/prompt registry | 11 | 运行时配置诊断、Embedding 参数适配和 Prompt registry 是否稳定 |
| async_ingestion_contract | PASS | RabbitMQ producer + ingestion worker contract | 14 | RabbitMQ 任务投递、worker 消费和入库任务错误处理是否稳定 |
