import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpenCheck, BrainCircuit, ChevronRight, FileText, FolderOpen, Globe2, LibraryBig,
  MessageSquarePlus, Network, Paperclip, Plus, RefreshCw, Search, Send,
  Settings2, Sparkles, Square, Trash2, Upload,
} from "lucide-react";
import { api, fileToBase64, normalizeApiUrl, streamChat } from "./api";
import PaperGraph from "./PaperGraph";
import { Reader } from "./Reader";

const NAV_ITEMS = [
  ["workspace", "研究工作台", MessageSquarePlus],
  ["library", "资料库", LibraryBig],
  ["paper-graph", "知识星图", Network],
  ["ingestion", "导入任务", Upload],
  ["settings", "连接设置", Settings2],
];

function IconButton({ label, children, ...props }) {
  return <button className="icon-button" title={label} aria-label={label} {...props}>{children}</button>;
}

function renderInlineMarkdown(text, keyPrefix) {
  const tokens = String(text || "").split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return tokens.map((token, index) => {
    const key = `${keyPrefix}-${index}`;
    if (token.startsWith("**") && token.endsWith("**")) return <strong key={key}>{token.slice(2, -2)}</strong>;
    if (token.startsWith("`") && token.endsWith("`")) return <code key={key}>{token.slice(1, -1)}</code>;
    return token;
  });
}

function AnswerMarkdown({ content, streaming }) {
  const lines = String(content || "").replace(/\r/g, "").split("\n");
  const blocks = [];
  let listItems = [];
  let listType = "ul";
  let listStart = 1;
  let orderedSequence = 0;
  let paragraph = [];
  let fencedCodeLanguage = null;
  let fencedCodeLines = [];

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ type: "paragraph", value: paragraph.join(" ") });
    paragraph = [];
  };
  const flushList = () => {
    if (listItems.length) blocks.push({ type: listType, items: listItems, start: listStart });
    listItems = [];
  };
  const flushFencedCode = () => {
    if (fencedCodeLanguage !== null) blocks.push({ type: "code", language: fencedCodeLanguage, value: fencedCodeLines.join("\n") });
    fencedCodeLanguage = null;
    fencedCodeLines = [];
  };

  const parseTableRow = (value) => value.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
  const isTableRow = (value) => /^\s*\|.*\|\s*$/.test(value) && (value.match(/\|/g) || []).length >= 2;
  const isTableSeparator = (value) => parseTableRow(value).every((cell) => /^:?-{3,}:?$/.test(cell));

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const rawLine = lines[lineIndex];
    const line = rawLine.trim();
    const codeFence = line.match(/^```([^`]*)$/);
    if (codeFence) {
      flushParagraph(); flushList();
      if (fencedCodeLanguage !== null) flushFencedCode();
      else fencedCodeLanguage = codeFence[1].trim();
      continue;
    }
    if (fencedCodeLanguage !== null) {
      fencedCodeLines.push(rawLine);
      continue;
    }
    const longInlineCode = line.match(/^`([\s\S]+)`$/);
    if (longInlineCode && longInlineCode[1].length >= 80) {
      flushParagraph(); flushList();
      blocks.push({ type: "code", language: "", value: longInlineCode[1] });
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const unordered = line.match(/^[-*]\s+(.+)$/);
    const normalizedOrderedLine = line.replace(/^\*\*(\d+[.)]\s+)(.*?)\*\*$/, "$1$2").replace(/^\*\*(\d+[.)]\s+)/, "$1");
    const ordered = normalizedOrderedLine.match(/^(\d+)[.)]\s+(.+)$/);
    if (!line) { flushParagraph(); flushList(); continue; }
    if (isTableRow(line)) {
      const tableLines = [line];
      while (lineIndex + 1 < lines.length && isTableRow(lines[lineIndex + 1].trim())) {
        tableLines.push(lines[lineIndex + 1].trim());
        lineIndex += 1;
      }
      if (tableLines.length >= 2 && isTableSeparator(tableLines[1])) {
        flushParagraph(); flushList();
        blocks.push({ type: "table", headers: parseTableRow(tableLines[0]), rows: tableLines.slice(2).filter((row) => !isTableSeparator(row)).map(parseTableRow) });
        continue;
      }
      paragraph.push(line);
      continue;
    }
    if (heading) {
      flushParagraph(); flushList();
      orderedSequence = 0;
      blocks.push({ type: `h${heading[1].length}`, value: heading[2] });
      continue;
    }
    if (/^(-{3,}|\*{3,})$/.test(line)) {
      flushParagraph(); flushList(); blocks.push({ type: "divider" }); continue;
    }
    if (unordered || ordered) {
      flushParagraph();
      const nextType = unordered ? "ul" : "ol";
      if (listItems.length && listType !== nextType) flushList();
      listType = nextType;
      if (ordered && !listItems.length) {
        const sourceIndex = Number(ordered[1] || 1);
        listStart = sourceIndex === 1 && orderedSequence > 0 ? orderedSequence + 1 : sourceIndex;
      }
      listItems.push((unordered || ordered)[2] || (unordered || ordered)[1]);
      if (ordered) orderedSequence = listStart + listItems.length - 1;
      continue;
    }
    flushList();
    paragraph.push(line);
  }
  flushParagraph(); flushList(); flushFencedCode();

  return <div className="answer-markdown">
    {blocks.map((block, index) => {
      const key = `${block.type}-${index}`;
      if (block.type === "divider") return <hr key={key} />;
      if (block.type === "h1") return <h3 key={key}>{renderInlineMarkdown(block.value, key)}</h3>;
      if (block.type === "h2") return <h4 key={key}>{renderInlineMarkdown(block.value, key)}</h4>;
      if (block.type === "h3") return <h5 key={key}>{renderInlineMarkdown(block.value, key)}</h5>;
      if (block.type === "code") return <figure className="answer-code-block" key={key}><figcaption>{block.language ? `代码片段 · ${block.language}` : "算法 / 伪代码片段"}</figcaption><pre><code>{block.value}</code></pre></figure>;
      if (block.type === "ul" || block.type === "ol") {
        const List = block.type;
        return <List key={key} start={block.type === "ol" ? block.start : undefined}>{block.items.map((item, itemIndex) => <li key={`${key}-${itemIndex}`}>{renderInlineMarkdown(item, `${key}-${itemIndex}`)}</li>)}</List>;
      }
      if (block.type === "table") return <div className="answer-table-wrap" key={key}><table className="answer-table"><thead><tr>{block.headers.map((cell, cellIndex) => <th key={`${key}-h-${cellIndex}`}>{renderInlineMarkdown(cell, `${key}-h-${cellIndex}`)}</th>)}</tr></thead><tbody>{block.rows.map((row, rowIndex) => <tr key={`${key}-r-${rowIndex}`}>{block.headers.map((_, cellIndex) => <td key={`${key}-r-${rowIndex}-c-${cellIndex}`}>{renderInlineMarkdown(row[cellIndex] || "", `${key}-r-${rowIndex}-c-${cellIndex}`)}</td>)}</tr>)}</tbody></table></div>;
      return <p key={key}>{renderInlineMarkdown(block.value, key)}</p>;
    })}
    {streaming && <span className="streaming-caret" aria-label="正在生成" />}
  </div>;
}

function Workspace({ state, setState, initialPrompt, clearInitialPrompt, onSessionChanged, onOpenDocument }) {
  const [input, setInput] = useState(""); const inputRef = useRef(null);
  const [visibleSources, setVisibleSources] = useState(null);
  const controllerRef = useRef(null);
  const workspaceBodyRef = useRef(null);
  const displayAnswer = (content) => {
    const legacySections = [
      "\n\n以下是基于检索证据的说明",
      "\n\n**证据分层说明**",
      "\n\n## 证据明确支持",
      "\n\n**重要说明**",
      "\n\n注：以上回答基于当前检索片段",
    ];
    return legacySections.reduce((answer, marker) => answer.split(marker)[0], String(content || "")).trim();
  };
  const openSource = (source) => {
    const metadata = source.metadata || {};
    const url = metadata.url || metadata.landing_page_url || metadata.pdf_url || (/^https?:\/\//.test(source.document_id || "") ? source.document_id : "");
    if (url) { window.open(url, "_blank", "noopener,noreferrer"); return; }
    if (source.document_id) onOpenDocument?.(source.document_id);
  };
  const submit = async (event) => {
    event?.preventDefault();
    const message = input.trim();
    if (!message || state.isStreaming) return;
    setInput("");
    const userMessage = { id: `user-${Date.now()}`, role: "user", content: message };
    const assistantMessage = { id: `assistant-${Date.now()}`, role: "assistant", content: "", sources: [] };
    setState((current) => ({ ...current, messages: [...current.messages, userMessage, assistantMessage], isStreaming: true, streamingMessageId: assistantMessage.id }));
    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      await streamChat(state.config.apiUrl, {
        message,
        session_id: state.activeSessionId,
        selected_document_ids: state.scopeMode === "selected_documents" ? state.selectedDocumentIds : [],
        use_react: state.config.useReact,
        use_web_search: state.config.allowWeb,
        metadata: {
          allow_openalex_search: state.config.allowOpenAlex,
          allow_web_search: state.config.allowWeb,
        },
      }, {
        signal: controller.signal,
        onEvent: (eventData) => {
          setState((current) => {
            const eventType = eventData.type || "";
            const messages = current.messages.map((item) => item.id !== assistantMessage.id ? item : {
              ...item,
              content: eventType === "text" ? `${item.content}${eventData.content || ""}` : item.content,
              sources: eventType === "sources" ? (eventData.sources || item.sources) : item.sources,
              status: eventType === "status" && eventData.user_visible !== false ? (eventData.content || item.status) : item.status,
              error: eventType === "error" ? (eventData.content || "请求失败") : item.error,
            });
            return { ...current, messages, activeSessionId: eventData.session_id || current.activeSessionId };
          });
        },
      });
    } catch (error) {
      if (error.name !== "AbortError") setState((current) => ({ ...current, messages: current.messages.map((item) => item.id === assistantMessage.id ? { ...item, content: `回答失败：${error.message}` } : item) }));
    } finally {
      controllerRef.current = null;
      setState((current) => ({ ...current, isStreaming: false, streamingMessageId: null }));
      onSessionChanged?.();
    }
  };
  useEffect(() => { if (initialPrompt) { setInput(initialPrompt); clearInitialPrompt(); requestAnimationFrame(() => inputRef.current?.focus()); } }, [initialPrompt, clearInitialPrompt]);
  useEffect(() => {
    const container = workspaceBodyRef.current;
    if (!container) return undefined;
    const frame = requestAnimationFrame(() => container.scrollTo({ top: container.scrollHeight, behavior: state.isStreaming ? "auto" : "smooth" }));
    return () => cancelAnimationFrame(frame);
  }, [state.messages, state.isStreaming]);
  return <main className="workspace">
    <header className="workspace-header"><div><span className="eyebrow">研究工作台</span><h2>论文知识库对话</h2></div></header>
    <section className="workspace-body" ref={workspaceBodyRef}>
      {!state.messages.length && <div className="empty-chat"><div className="workspace-mark"><BookOpenCheck size={24} /></div><h1>从论文知识库开始研究</h1><p>选择一篇、多篇或整个知识库进行提问，回答会回到可追溯的论文证据。</p></div>}
      <div className="message-list">{state.messages.map((message) => <article className={`message message-${message.role}`} key={message.id}><div className="message-avatar">{message.role === "user" ? "我" : <BookOpenCheck size={16} />}</div><div className="message-content">{message.role === "assistant" && <div className="message-label">PaperWeave</div>}{message.status && !message.content ? <p className="retrieval-status">{message.status}</p> : null}<div className="message-text"><AnswerMarkdown content={displayAnswer(message.content) || (message.error ? `回答失败：${message.error}` : state.isStreaming && message.id === state.streamingMessageId ? "正在生成..." : "")} streaming={state.isStreaming && message.id === state.streamingMessageId} /></div>{message.sources?.length ? <button type="button" className="evidence-summary" onClick={() => setVisibleSources(visibleSources === message.id ? null : message.id)}>{visibleSources === message.id ? "收起依据" : `查看依据 (${message.sources.length})`}</button> : null}{visibleSources === message.id ? <div className="evidence-drawer">{message.sources.map((source, index) => { const metadata = source.metadata || {}; const isWeb = Boolean(metadata.url || metadata.landing_page_url || metadata.pdf_url || /^https?:\/\//.test(source.document_id || "")); return <article key={`${source.chunk_id || source.document_id || "source"}-${index}`}><div><span>{source.source_type === "web" ? "网页" : source.source_type === "artifact" ? "图表/算法" : "本地论文"}</span><strong>{source.document_title || source.document_source || "未命名来源"}</strong>{source.snippet ? <p>{source.snippet}</p> : null}</div><button type="button" onClick={() => openSource(source)} disabled={!isWeb && !source.document_id}>{isWeb ? "打开" : "论文"}</button></article>; })}</div> : null}</div></article>)}</div>
    </section>
    <form className="composer-wrap" onSubmit={submit}><div className="composer-tools"><button className={`tool-toggle ${state.config.useReact ? "enabled" : ""}`} type="button" title="启用多步规划、检索和答案审阅" onClick={() => setState((current) => ({ ...current, config: { ...current.config, useReact: !current.config.useReact } }))}><Sparkles size={14} />深度分析</button><button className={`tool-toggle ${state.config.allowOpenAlex ? "enabled" : ""}`} type="button" disabled={!state.capabilities?.openalex} title={state.capabilities?.openalex ? "按需检索 OpenAlex 学术元数据" : "OpenAlex 服务当前不可用"} onClick={() => setState((current) => ({ ...current, config: { ...current.config, allowOpenAlex: !current.config.allowOpenAlex } }))}><BrainCircuit size={14} />OpenAlex</button><button className={`tool-toggle ${state.config.allowWeb ? "enabled" : ""}`} type="button" disabled={!state.capabilities?.web} title={state.capabilities?.web ? "按需检索外部网页" : "网页检索服务当前不可用"} onClick={() => setState((current) => ({ ...current, config: { ...current.config, allowWeb: !current.config.allowWeb } }))}><Globe2 size={14} />网页检索</button></div><div className="composer"><textarea ref={inputRef} value={input} onChange={(event) => setInput(event.target.value)} placeholder="提出一个关于论文、方法或研究方向的问题..." onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) submit(event); }} /><div className="composer-bottom"><span>Enter 发送，Shift + Enter 换行</span>{state.isStreaming ? <button className="send-button stop" type="button" onClick={() => controllerRef.current?.abort()} title="停止生成"><Square size={13} /></button> : <button className="send-button" type="submit" disabled={!input.trim()} title="发送"><Send size={16} /></button>}</div></div></form>
  </main>;
}

function Library({ state, setState, navigate, openDocument }) {
  const [query, setQuery] = useState("");
  const docs = useMemo(() => state.documents.filter((doc) => `${doc.title || ""} ${doc.source || ""}`.toLowerCase().includes(query.toLowerCase())), [state.documents, query]);
  const remove = async (doc) => {
    if (!window.confirm(`删除《${doc.title}》及其所有知识证据？`)) return;
    await api.deleteDocument(state.config.apiUrl, doc.id);
    const result = await api.documents(state.config.apiUrl);
    setState((current) => ({ ...current, documents: result.documents || [] }));
  };
  return <main className="content-page"><header className="page-header"><div><span className="eyebrow">本地知识库</span><h2>资料库</h2></div><button className="primary-button" onClick={() => navigate("ingestion")}><Upload size={16} />导入 PDF</button></header><div className="filter-row"><label><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="筛选标题或来源" /></label><span>{docs.length} 篇文档</span></div><section className="document-list">{docs.length ? docs.map((doc) => <article className="document-row" key={doc.id}><div className="document-icon"><FileText size={19} /></div><button className="document-open" onClick={() => openDocument(doc)}><strong>{doc.title || "未命名论文"}</strong><p>{doc.source || "本地导入"} · {doc.chunk_count ?? 0} 个检索片段</p></button><div className="document-actions"><button onClick={() => navigate("workspace", `请围绕《${doc.title}》进行分析。`)}>分析<ChevronRight size={16} /></button><IconButton label="删除论文" onClick={() => remove(doc)}><Trash2 size={15} /></IconButton></div></article>) : <div className="page-empty"><FolderOpen size={28} />资料库为空或没有匹配文档。</div>}</section></main>;
}

function Ingestion({ state, setState, onIngestionDone }) {
  const [files, setFiles] = useState([]); const [mode, setMode] = useState("full"); const [jobs, setJobs] = useState([]); const [notice, setNotice] = useState(""); const [submitting, setSubmitting] = useState(false);
  const refreshJobs = useCallback(() => api.ingestionTasks(state.config.apiUrl).then(setJobs).catch((error) => setNotice(`无法读取入库队列：${error.message}`)), [state.config.apiUrl]);
  useEffect(() => { refreshJobs(); const timer = setInterval(refreshJobs, 1800); return () => clearInterval(timer); }, [refreshJobs]);
  useEffect(() => { if (!jobs.some((job) => job.status === "done")) return; api.documents(state.config.apiUrl).then(({ documents }) => setState((current) => ({ ...current, documents: documents || [] }))).catch(() => {}); onIngestionDone(); }, [jobs, onIngestionDone, setState, state.config.apiUrl]);
  const submit = async () => {
    if (!files.length) return;
    setSubmitting(true);
    setNotice("");
    let submitted = 0;
    const failed = [];
    for (const file of files) {
      try {
        const payload = { filename: file.name, content_base64: await fileToBase64(file), fast: mode === "fast" };
        await api.uploadFile(state.config.apiUrl, payload);
        submitted += 1;
        setNotice(`已提交 ${submitted}/${files.length} 篇，正在继续加入队列...`);
        refreshJobs();
      } catch (error) {
        failed.push(`${file.name}：${error.message}`);
      }
    }
    setFiles([]);
    setNotice(failed.length
      ? `已提交 ${submitted}/${files.length} 篇。未提交：${failed.join("；")}`
      : `${submitted} 篇论文已加入入库队列。`);
    refreshJobs();
    setSubmitting(false);
  };
  const control = async (action, taskId) => { try { await action(state.config.apiUrl, taskId); refreshJobs(); } catch (error) { setNotice(`操作失败：${error.message}`); } };
  const move = async (taskId, direction) => {
    const reorderable = jobs.filter((job) => ["queued", "paused", "paused_quota"].includes(job.status));
    const index = reorderable.findIndex((job) => job.task_id === taskId);
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= reorderable.length) return;
    [reorderable[index], reorderable[nextIndex]] = [reorderable[nextIndex], reorderable[index]];
    try { await api.reorderIngestionTasks(state.config.apiUrl, reorderable.map((job) => job.task_id)); refreshJobs(); } catch (error) { setNotice(`调整队列顺序失败：${error.message}`); }
  };
  const active = jobs.filter((job) => ["queued", "processing", "paused", "paused_quota"].includes(job.status));
  return <main className="content-page"><header className="page-header"><div><span className="eyebrow">知识库构建</span><h2>导入任务</h2></div><span className="selection-count">{active.length} 个待处理</span></header><section className="upload-layout"><div className="upload-zone"><Upload size={28} /><strong>{files.length ? `已选择 ${files.length} 篇 PDF` : "选择一篇或多篇 PDF 论文"}</strong><p>每篇论文会成为独立的持久化任务，由 worker 顺序入库。</p><label className="primary-button"><Paperclip size={16} />选择 PDF<input type="file" accept="application/pdf" multiple onChange={(event) => setFiles(Array.from(event.target.files || []))} /></label></div><div className="ingestion-options"><span className="option-label">入库策略</span><div className="ingestion-mode-grid"><button className={`ingestion-mode ${mode === "fast" ? "selected" : ""}`} onClick={() => setMode("fast")}><strong>快速入库</strong><span>仅论文正文</span></button><button className={`ingestion-mode ${mode === "full" ? "selected" : ""}`} onClick={() => setMode("full")}><strong>完整入库</strong><span>正文、算法、表格、图片</span></button></div><button className="primary-button" disabled={!files.length || !state.online || submitting} onClick={submit}><Upload size={16} />{submitting ? "正在提交..." : `加入队列${files.length ? ` (${files.length})` : ""}`}</button></div></section>{notice && <p className="notice">{notice}</p>}<section className="ingestion-queue"><div className="queue-heading"><div><span className="eyebrow">任务队列</span><h3>入库进度</h3></div><button className="text-button" onClick={refreshJobs}><RefreshCw size={14} />刷新</button></div>{jobs.length ? jobs.map((job) => { const activeJob = ["queued", "processing"].includes(job.status); const paused = ["paused", "paused_quota"].includes(job.status); const status = { queued: "等待中", processing: "处理中", paused: "已暂停", paused_quota: "额度暂停", done: "已完成", failed: "失败" }[job.status] || job.status; const reorderable = ["queued", "paused", "paused_quota"].includes(job.status); return <article className={`ingestion-task status-${job.status}`} key={job.task_id}><div className="task-order"><button type="button" disabled={!reorderable} title="上移任务" onClick={() => move(job.task_id, -1)}>↑</button><button type="button" disabled={!reorderable} title="下移任务" onClick={() => move(job.task_id, 1)}>↓</button></div><div className="task-main"><div className="task-title"><strong>{job.filename}</strong><span className={`status-pill status-${job.status}`}>{status}</span></div><div className="progress-track"><i style={{ width: `${job.progress_percent || 0}%` }} /></div><small>{job.progress_percent || 0}% · {job.progress_stage || "等待入库"}{job.fast ? " · 快速模式" : " · 完整模式"}</small>{job.error_message && <p className="task-error">{job.error_message}</p>}</div><div className="task-actions">{activeJob && <IconButton label="暂停任务" onClick={() => control(api.pauseIngestionTask, job.task_id)}><Square size={15} /></IconButton>}{paused && <IconButton label="恢复任务" onClick={() => control(api.resumeIngestionTask, job.task_id)}><RefreshCw size={15} /></IconButton>}{job.status !== "done" && <IconButton label="删除任务" onClick={() => control(api.deleteIngestionTask, job.task_id)}><Trash2 size={15} /></IconButton>}</div></article>; }) : <div className="page-empty"><Upload size={28} />暂时没有入库任务。</div>}</section></main>;
}

function Settings({ state, setState, refresh }) {
  const [url, setUrl] = useState(state.config.apiUrl); const [message, setMessage] = useState("");
  const save = async () => { const apiUrl = normalizeApiUrl(url); localStorage.setItem("agentic-rag-api-url", apiUrl); setState((current) => ({ ...current, config: { ...current.config, apiUrl } })); setMessage("已保存，正在重新连接..."); await refresh(apiUrl); };
  return <main className="content-page narrow"><header className="page-header"><div><span className="eyebrow">运行环境</span><h2>连接设置</h2></div></header><section className="settings-group"><label>Agent API 地址<input value={url} onChange={(event) => setUrl(event.target.value)} /></label><button className="primary-button" onClick={save}><RefreshCw size={16} />保存并检测</button>{message && <p className="notice">{message}</p>}</section><section className="status-grid"><div><span>API 服务</span><strong className={state.online ? "positive" : "negative"}>{state.online ? "已连接" : "未连接"}</strong></div><div><span>论文资料库</span><strong>{state.documents.length} 篇</strong></div><div><span>当前地址</span><strong>{normalizeApiUrl(url).replace(/^https?:\/\//, "")}</strong></div></section><p className="settings-hint"><Settings2 size={15} />修改地址后会立即重新检测连接状态。</p></main>;
}

export default function App() {
  const [page, setPage] = useState("workspace"); const [pendingPrompt, setPendingPrompt] = useState(""); const [readerDocument, setReaderDocument] = useState(null); const [paperGraph, setPaperGraph] = useState({ nodes: [], edges: [], version: 0 });
  const [state, setState] = useState(() => { const stored = localStorage.getItem("agentic-rag-api-url"); return { config: { apiUrl: stored === "http://localhost:8000" ? "http://localhost:8059" : stored || "http://localhost:8059", useReact: true, allowOpenAlex: false, allowWeb: false }, online: false, documents: [], sessions: [], messages: [], activeSessionId: null, selectedDocumentIds: [], scopeMode: "knowledge_base", isStreaming: false, streamingMessageId: null }; });
  const refreshPaperGraph = useCallback(async (url = state.config.apiUrl) => { const graph = await api.paperGraph(url); setPaperGraph((current) => current.version === graph.version ? current : graph); }, [state.config.apiUrl]);
  const refreshSessions = useCallback(async (url = state.config.apiUrl) => {
    const result = await api.sessions(url);
    setState((current) => ({ ...current, sessions: result.sessions || [] }));
  }, [state.config.apiUrl]);
  const refresh = useCallback(async (url = state.config.apiUrl) => { try { await api.health(url); const [docs, sessions, capabilities] = await Promise.all([api.documents(url), api.sessions(url), api.capabilities(url)]); setState((current) => ({ ...current, online: true, documents: docs.documents || [], sessions: sessions.sessions || [], capabilities })); refreshPaperGraph(url).catch(() => {}); } catch { setState((current) => ({ ...current, online: false })); } }, [refreshPaperGraph, state.config.apiUrl]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { if (page !== "paper-graph" || !state.online) return undefined; const timer = setInterval(() => refreshPaperGraph().catch(() => {}), 12000); return () => clearInterval(timer); }, [page, refreshPaperGraph, state.online]);
  const navigate = (target, prompt = "") => { setPage(target); setPendingPrompt(prompt); };
  const createSession = () => { setPendingPrompt(""); setPage("workspace"); setState((current) => ({ ...current, messages: [], activeSessionId: null })); };
  const openSession = async (sessionId) => {
    if (!sessionId || state.isStreaming) return;
    try {
      const result = await api.sessionMessages(state.config.apiUrl, sessionId);
      setPage("workspace");
      setState((current) => ({ ...current, activeSessionId: sessionId, messages: (result.messages || []).map((message) => ({ id: message.message_id, role: message.role, content: message.content, sources: message.metadata?.sources || [] })) }));
    } catch (error) { window.alert(`无法恢复会话：${error.message}`); }
  };
  const removeSession = async (event, sessionId) => {
    event.stopPropagation();
    if (!window.confirm("删除此会话及其记忆快照？")) return;
    try {
      await api.deleteSession(state.config.apiUrl, sessionId);
      setState((current) => ({ ...current, sessions: current.sessions.filter((session) => session.session_id !== sessionId), ...(current.activeSessionId === sessionId ? { activeSessionId: null, messages: [] } : {}) }));
    } catch (error) { window.alert(`删除会话失败：${error.message}`); }
  };
  const openEvidenceDocument = (documentId) => {
    const document = state.documents.find((item) => item.id === documentId);
    if (document) { setReaderDocument(document); setPage("reader"); }
  };
  const content = page === "workspace" ? <Workspace state={state} setState={setState} initialPrompt={pendingPrompt} clearInitialPrompt={() => setPendingPrompt("")} onSessionChanged={() => refreshSessions().catch(() => {})} onOpenDocument={openEvidenceDocument} /> : page === "library" ? <Library state={state} setState={setState} navigate={navigate} openDocument={(doc) => { setReaderDocument(doc); setPage("reader"); }} /> : page === "paper-graph" ? <PaperGraph graph={paperGraph} onClose={() => setPage("workspace")} onAsk={(node) => navigate("workspace", `请围绕《${node.title}》结合本地知识库进行分析。`)} /> : page === "reader" && readerDocument ? <Reader document={readerDocument} state={state} setState={setState} onBack={() => setPage("library")} onAskSelection={(selection) => navigate("workspace", selection)} /> : page === "ingestion" ? <Ingestion state={state} setState={setState} onIngestionDone={refreshPaperGraph} /> : <Settings state={state} setState={setState} refresh={refresh} />;
  return <div className="app-shell"><aside className="sidebar"><div className="brand"><div className="brand-mark"><BookOpenCheck size={19} /></div><div><strong>PaperWeave</strong><span>Research Knowledge Base</span></div></div><button className="new-session" onClick={createSession}><Plus size={17} />新建对话</button><nav>{NAV_ITEMS.map(([key, label, Icon]) => <button key={key} className={page === key ? "nav-active" : ""} onClick={() => setPage(key)}><Icon size={17} />{label}</button>)}</nav><section className="session-section"><div className="section-title"><span>最近对话</span><span>{state.sessions.length}</span></div><div className="session-list">{state.sessions.map((session) => <div className={`session-entry ${state.activeSessionId === session.session_id ? "session-active" : ""}`} key={session.session_id}><button type="button" onClick={() => openSession(session.session_id)} title={session.title || "未命名对话"}><span>{session.title || "未命名对话"}</span><small>{session.last_message_preview || "暂无消息"}</small></button><IconButton label="删除会话" onClick={(event) => removeSession(event, session.session_id)}><Trash2 size={14} /></IconButton></div>)}</div></section></aside>{content}</div>;
}
