const DEFAULT_API_URL = import.meta.env.VITE_API_URL || "http://localhost:8059";

export function normalizeApiUrl(value) {
  const base = (value || DEFAULT_API_URL).trim();
  return (base.includes("://") ? base : `http://${base}`).replace(/\/$/, "");
}

async function request(baseUrl, path, options = {}) {
  let response;
  try {
    response = await fetch(`${normalizeApiUrl(baseUrl)}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (error) {
    throw new Error(`无法连接 API（${normalizeApiUrl(baseUrl)}）：${error.message}`);
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* use status */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}

export const api = {
  health: (baseUrl) => request(baseUrl, "/health/live"),
  documents: (baseUrl) => request(baseUrl, "/documents?limit=200&offset=0"),
  paperGraph: (baseUrl) => request(baseUrl, "/paper-graph"),
  artifact: (baseUrl, id) => request(baseUrl, `/artifacts/${id}`),
  artifactImageUrl: (baseUrl, id) => `${normalizeApiUrl(baseUrl)}/artifacts/${id}/image`,
  documentPdfUrl: (baseUrl, id) => `${normalizeApiUrl(baseUrl)}/documents/${id}/pdf`,
  translateDocument: (baseUrl, id, language) => request(baseUrl, `/documents/${id}/translations/${language}`, { method: "POST" }),
  translateSelection: (baseUrl, id, language, payload) => request(baseUrl, `/documents/${id}/selection-translations/${language}`, { method: "POST", body: JSON.stringify(payload) }),
  annotations: (baseUrl, id) => request(baseUrl, `/documents/${id}/annotations`),
  addAnnotation: (baseUrl, id, payload) => request(baseUrl, `/documents/${id}/annotations`, { method: "POST", body: JSON.stringify(payload) }),
  updateAnnotation: (baseUrl, id, annotationId, payload) => request(baseUrl, `/documents/${id}/annotations/${annotationId}`, { method: "PATCH", body: JSON.stringify(payload) }),
  deleteAnnotation: (baseUrl, id, annotationId) => request(baseUrl, `/documents/${id}/annotations/${annotationId}`, { method: "DELETE" }),
  deleteDocument: (baseUrl, id) => request(baseUrl, `/documents/${id}`, { method: "DELETE" }),
  upgradeDocument: (baseUrl, id) => request(baseUrl, `/documents/${id}/upgrade-full`, { method: "POST" }),
  sessions: (baseUrl) => request(baseUrl, "/sessions?limit=20&days=7"),
  sessionMessages: (baseUrl, id) => request(baseUrl, `/sessions/${id}/messages`),
  sessionMemory: (baseUrl, id) => request(baseUrl, `/sessions/${id}/memory`),
  deleteSession: (baseUrl, id) => request(baseUrl, `/sessions/${id}`, { method: "DELETE" }),
  capabilities: async (baseUrl) => {
    const [openalex, web] = await Promise.allSettled([
      request(baseUrl, "/openalex/status"), request(baseUrl, "/web-search/status"),
    ]);
    return {
      openalex: openalex.status === "fulfilled" && Boolean(openalex.value.enabled),
      web: web.status === "fulfilled" && Boolean(web.value.enabled),
      webProvider: web.status === "fulfilled" ? web.value.provider : "",
    };
  },
  ingestionTasks: (baseUrl) => request(baseUrl, "/ingestion/tasks"),
  uploadBatch: (baseUrl, files) => request(baseUrl, "/ingestion/task-batches", { method: "POST", body: JSON.stringify({ files }) }),
  uploadFile: (baseUrl, file) => request(baseUrl, "/ingestion/task-batches", { method: "POST", body: JSON.stringify({ files: [file] }) }),
  resumeIngestionTask: (baseUrl, id) => request(baseUrl, `/ingestion/tasks/${id}/resume`, { method: "POST" }),
  pauseIngestionTask: (baseUrl, id) => request(baseUrl, `/ingestion/tasks/${id}/pause`, { method: "POST" }),
  deleteIngestionTask: (baseUrl, id) => request(baseUrl, `/ingestion/tasks/${id}`, { method: "DELETE" }),
  reorderIngestionTasks: (baseUrl, taskIds) => request(baseUrl, "/ingestion/tasks/order", { method: "PUT", body: JSON.stringify({ task_ids: taskIds }) }),
  cancelStream: (baseUrl, runId) => request(baseUrl, `/chat/stream/${runId}/cancel`, { method: "POST" }),
};

export async function streamChat(baseUrl, payload, { signal, onEvent }) {
  const response = await fetch(`${normalizeApiUrl(baseUrl)}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok || !response.body) {
    let detail = "";
    try {
      const body = await response.json();
      detail = typeof body?.detail === "string" ? body.detail : body?.error || "";
    } catch { /* keep the HTTP fallback */ }
    if (response.status === 429) {
      throw new Error("请求过于频繁，请稍后重试。");
    }
    throw new Error(detail || `流式请求失败：HTTP ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try { onEvent(JSON.parse(line.slice(6))); } catch { /* skip malformed SSE frame */ }
      }
    }
  }
}

export async function streamDocumentTranslation(baseUrl, documentId, language, { signal, onEvent }) {
  const response = await fetch(`${normalizeApiUrl(baseUrl)}/documents/${documentId}/translations/${language}/stream`, {
    method: "POST",
    headers: { Accept: "text/event-stream" },
    signal,
  });
  if (!response.ok || !response.body) throw new Error(`翻译流请求失败：HTTP ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const line = frame.split("\n").find((entry) => entry.startsWith("data: "));
      if (!line) continue;
      let event;
      try { event = JSON.parse(line.slice(6)); } catch { continue; }
      if (event.type === "error") throw new Error(event.content || "翻译服务返回错误");
      onEvent(event);
    }
  }
}

export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("无法读取文件"));
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.readAsDataURL(file);
  });
}
