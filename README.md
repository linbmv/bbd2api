# bbd2api

A lightweight Flask reverse proxy that bridges the **Claude Messages API** and **OpenAI Chat Completions API** to [backboard.io](https://app.backboard.io)'s assistant/thread API — with tool calling, thread pooling, and sub-5s response times.

## Features

- **Dual API compatibility** — `/v1/messages` (Claude) and `/v1/chat/completions` (OpenAI)
- **Tool calling** — prompt-injection approach; compact format (~800 chars vs ~68KB for full schemas)
- **Multi-turn tool continuity** — `tool_result` messages route back to the correct thread automatically
- **Thread pool** — pre-warms N threads at startup; zero cold-start on new conversations
- **Global assistant** — single assistant reused across all requests; eliminates per-request creation overhead
- **System prompt stability** — strips dynamic tags (`<system-reminder>` etc.) before hashing to prevent cache misses
- **Model routing** — auto-detects provider from model name (claude→anthropic, gpt→openai, gemini→google, etc.)

## Quick Start

```bash
git clone https://github.com/N1nEmAn/bbd2api.git
cd bbd2api
pip install -r requirements.txt
cp .env.example .env
# edit .env and set BBD_API_KEY
python server.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BBD_API_KEY` | *(required)* | API key(s) — comma-separated for multi-key rotation: `key1,key2,key3` |
| `BBD_UPSTREAM` | `https://app.backboard.io/api` | Upstream API base URL |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `10088` | Listen port |
| `THREAD_TTL` | `1800` | Thread cache TTL in seconds |
| `DEBUG` | `1` | Enable debug logging (`0` to disable) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/messages` | Claude Messages API (streaming + sync) |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions API |
| `GET` | `/v1/models` | List available models |
| `GET` | `/health` | Health check |
| `GET` | `/debug/state` | Show assistant/thread/pool state |
| `POST` | `/debug/clear` | Clear thread cache |

## Usage with Claude Code / OpenAI clients

```bash
# Claude-style
curl http://localhost:10088/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# OpenAI-style
curl http://localhost:10088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

Or point any OpenAI-compatible client (LiteLLM, Continue, etc.) at `http://localhost:10088`.

## Architecture

```
Client (Claude/OpenAI API)
        │
        ▼
   bbd2api proxy (Flask, port 10088)
        │  • thread pool (pre-warmed)
        │  • tool prompt injection
        │  • tool_result continuity (tool_use_id → thread mapping)
        │  • SSE format translation
        ▼
  backboard.io API
  /assistants/{aid}/threads/{tid}/messages
```

### Tool Calling Flow

1. Client sends request with `tools` list
2. Proxy injects compact tool format instructions into the user message
3. Model responds with JSON: `{"tool":"name","args":{...}}`
4. Proxy translates to `tool_use` content blocks, stores `tool_use_id → thread_id`
5. Client executes tool, sends back `tool_result`
6. Proxy detects `tool_result`, looks up thread via `tool_use_id`, sends result to same thread
7. Model continues reasoning and produces final answer

## License

[CC BY-NC-SA 4.0](LICENSE) — Non-commercial use only. If you use or adapt this project, **you must credit the original author (N1nEmAn)** and distribute your changes under the same license.

Commercial use is **prohibited** without explicit written permission from the author.

## Disclaimer

See [DISCLAIMER.md](DISCLAIMER.md).

---

## Community

The author endorses and supports the **[LINUX DO](https://linux.do)** community.

---

# bbd2api（中文文档）

将 **Claude Messages API** 和 **OpenAI Chat Completions API** 转接到 [backboard.io](https://app.backboard.io) 助手/线程 API 的轻量级 Flask 反向代理，支持工具调用、线程池预热，响应时间低于 5 秒。

## 功能特性

- **双 API 兼容** — `/v1/messages`（Claude）和 `/v1/chat/completions`（OpenAI）
- **工具调用** — prompt 注入方案，极简格式（~800 字符 vs 完整 schema 的 ~68KB）
- **多轮工具连续性** — `tool_result` 消息自动路由回正确的 thread，对话不断链
- **线程池** — 启动时预热 N 个 thread，新对话零冷启动
- **全局 assistant** — 单一 assistant 复用，消除每次请求创建 assistant 的开销
- **系统提示稳定性** — hash 前剥离动态标签（`<system-reminder>` 等），防止缓存失效
- **模型路由** — 根据模型名自动识别 provider（claude→anthropic，gpt→openai，gemini→google 等）

## 快速开始

```bash
git clone https://github.com/N1nEmAn/bbd2api.git
cd bbd2api
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 BBD_API_KEY
python server.py
```

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BBD_API_KEY` | *（必填）* | API Key，逗号分隔多个 Key 可启用轮换：`key1,key2,key3` |
| `BBD_UPSTREAM` | `https://app.backboard.io/api` | 上游 API 基础地址 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `10088` | 监听端口 |
| `THREAD_TTL` | `1800` | Thread 缓存过期时间（秒） |
| `DEBUG` | `1` | 开启调试日志（`0` 关闭） |

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/messages` | Claude Messages API（流式 + 同步） |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions API |
| `GET` | `/v1/models` | 获取可用模型列表 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/debug/state` | 查看 assistant/thread/pool 状态 |
| `POST` | `/debug/clear` | 清除 thread 缓存 |

## 调用示例

```bash
# Claude 风格
curl http://localhost:10088/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"你好"}],"stream":true}'

# OpenAI 风格
curl http://localhost:10088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"你好"}]}'
```

也可以将任意兼容 OpenAI 的客户端（LiteLLM、Continue 等）指向 `http://localhost:10088`。

## 架构说明

```
客户端（Claude/OpenAI API）
        │
        ▼
   bbd2api 代理（Flask，端口 10088）
        │  • 线程池（预热）
        │  • 工具 prompt 注入
        │  • tool_result 续接（tool_use_id → thread 映射）
        │  • SSE 格式转换
        ▼
  backboard.io API
  /assistants/{aid}/threads/{tid}/messages
```

### 工具调用流程

1. 客户端发送含 `tools` 的请求
2. 代理将极简工具格式说明注入用户消息
3. 模型输出 JSON：`{"tool":"name","args":{...}}`
4. 代理转换为 `tool_use` content block，并记录 `tool_use_id → thread_id`
5. 客户端执行工具，返回 `tool_result`
6. 代理检测到 `tool_result`，通过 `tool_use_id` 找回原 thread，将结果发送进去
7. 模型基于结果继续推理，输出最终回答

### 多 Key 轮换

```
BBD_API_KEY=key1,key2,key3
```

- **Round-robin** 轮换：每次请求取下一个 key
- **自动跳过**：某个 key 连续失败 3 次暂时跳过
- **自动恢复**：成功一次即重置失败计数

## 许可证

[CC BY-NC-SA 4.0](LICENSE) — 仅限非商业用途。使用或改编本项目时，**必须注明原作者（N1nEmAn）**，并以相同许可证分发衍生作品。

未经作者书面许可，**禁止商业用途**。

## 免责声明

见 [DISCLAIMER.md](DISCLAIMER.md)。

---

## 社区

作者认可并支持 **[LINUX DO](https://linux.do)** 社区。

