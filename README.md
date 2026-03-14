# YouTube 一键总结工具

本地运行 · 零 API 费用 · Ollama 驱动

---

## 快速启动

### 1. 安装 Python 依赖

```bash
cd youtube-summarizer
pip install -r requirements.txt
```

### 2. 确认 Ollama 已运行

```bash
# 如果还没安装 Ollama：
# Windows: 去 https://ollama.com 下载安装包
# Linux/Mac: curl -fsSL https://ollama.ai/install.sh | sh

# 拉取模型（首次需要，约 5-6GB）
ollama pull qwen3.5:9b

# 启动 Ollama（如果没有自动启动）
ollama serve
```

### 3. 启动后端

```bash
cd backend
uvicorn main:app --reload --port 8000
```

看到 `Uvicorn running on http://127.0.0.1:8000` 即成功。

### 4. 打开前端

浏览器访问：**http://127.0.0.1:8000**

或直接用浏览器打开 `frontend/index.html`（需要后端已启动）。

---

## 使用

1. 粘贴 YouTube 链接（支持自动触发）
2. 选择 Ollama 模型（默认 qwen3.5:9b）
3. 点击「一键总结」或按 Enter
4. 等待约 30-60 秒（取决于视频长度和模型大小）
5. 左侧查看 AI 总结，右侧查看双语字幕

---

## 常见问题

**Q：字幕获取失败？**
- 部分视频没有字幕（直播回放、纯音乐视频等）
- 如果在中国大陆，可能需要代理访问 YouTube CDN

**Q：Ollama 连接失败？**
- 确认运行了 `ollama serve`
- 确认端口 11434 未被占用

**Q：总结很慢？**
- 9B 模型在 3090Ti 上约 30-60 秒
- 可以换更小的模型（llama3.1:8b 更快）

---

## 项目结构

```
youtube-summarizer/
├── backend/
│   ├── main.py          # FastAPI 入口
│   ├── transcript.py    # 字幕提取
│   ├── summarizer.py    # AI 总结
│   └── translator.py    # 批量翻译
├── frontend/
│   └── index.html       # 单页前端
├── cache/               # 本地缓存（自动创建）
└── requirements.txt
```
