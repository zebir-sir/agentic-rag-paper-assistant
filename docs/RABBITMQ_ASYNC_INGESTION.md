# RabbitMQ 异步文档入库

## 为什么要做异步入库
- PDF 解析、切块、embedding、写库耗时较长。
- 异步化后，上传请求快速返回 `task_id`，前端可轮询任务状态，不阻塞用户继续操作。

## 整体流程
1. 用户上传 PDF（异步模式）。
2. API 创建 `ingestion_tasks` 记录（`queued`）。
3. API 向 RabbitMQ 主队列投递任务消息。
4. `ingestion-worker` 消费消息并执行现有入库流程。
5. worker 更新任务状态：
   - `queued -> processing -> done`
   - 或失败时 `queued -> processing -> queued(retry) -> failed`

## 接口
- `POST /ingestion/tasks`
- `GET /ingestion/tasks/{task_id}`

## 服务
- `rabbitmq`
- `ingestion-worker`
- `api`
- `postgres`
- `ui`

## 环境变量
- `RABBITMQ_URL`
- `INGESTION_QUEUE_NAME`
- `INGESTION_DLQ_NAME`
- `INGESTION_MAX_RETRIES`

## 启动方式
```bash
docker compose up -d postgres rabbitmq api ingestion-worker ui
```

## 验证方式
1. 在 UI 上传 PDF，选择异步入库。
2. 确认返回并展示 `task_id`。
3. 观察任务状态变化：
   - `queued -> processing -> done`
   - 或失败链路：`queued -> processing -> queued/retry -> failed`
4. 在 RabbitMQ 管理台查看队列：
   - [http://localhost:15672](http://localhost:15672)
5. 数据库验证：
   - `ingestion_tasks` 状态与 `retry_count`、`error_message` 正确
   - `documents` 有新增文档
   - `chunks` 有新增分块

## 当前限制
- 已实现基础幂等保护。
- 跨系统事务级幂等暂未彻底解决（例如“写入成功但状态更新前崩溃”场景）。
- `document_id` 回填仍可能依赖现有入库链路返回方式（当前实现可能走日志提取）。
