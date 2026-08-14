import { useEffect, useRef, useState } from "react";
import { BookOpen, FileText, LoaderCircle, MessageSquare, Minus, Plus, RotateCcw, Trash2, X } from "lucide-react";
import * as pdfjsLib from "pdfjs-dist";
import PdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?worker";
import { api } from "./api";

pdfjsLib.GlobalWorkerOptions.workerPort = new PdfWorker();

function Marker({ item, pageRef, onMove, onOpen }) {
  const drag = useRef(null);
  const [position, setPosition] = useState({ x: item.page_x, y: item.page_y });
  useEffect(() => setPosition({ x: item.page_x, y: item.page_y }), [item.page_x, item.page_y]);
  const down = (event) => { event.stopPropagation(); drag.current = { moved: false }; event.currentTarget.setPointerCapture(event.pointerId); };
  const move = (event) => { if (!drag.current) return; const rect = pageRef.current?.getBoundingClientRect(); if (!rect) return; const next = { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) }; drag.current = { moved: true, position: next }; setPosition(next); };
  const up = (event) => { const result = drag.current; drag.current = null; event.currentTarget.releasePointerCapture?.(event.pointerId); if (!result) return; if (result.moved) onMove(item, result.position); else onOpen(item); };
  return <button className="annotation-marker" style={{ left: `${position.x * 100}%`, top: `${position.y * 100}%`, background: item.color || "#e0b84c" }} aria-label="批注标记" onPointerDown={down} onPointerMove={move} onPointerUp={up} />;
}

function PdfPage({ page, rasterUrl, annotations, zoom, adding, draft, onPlace, onMove, onSaveDraft, onCancelDraft, onTextSelected, onDelete, onAskSelection, activeSelectionPage, onSelectionActive, onRenderError }) {
  const pageRef = useRef(null); const rasterRef = useRef(null); const textRef = useRef(null); const pageTextRef = useRef(""); const selectionRectsRef = useRef([]); const selectionJustCreatedRef = useRef(false); const [opened, setOpened] = useState(null); const [selectionAction, setSelectionAction] = useState(null); const [pageSize, setPageSize] = useState(null);
  const highlightName = "reader-selection";
  useEffect(() => () => { if (typeof CSS !== "undefined" && CSS.highlights) CSS.highlights.delete(highlightName); }, []);
  useEffect(() => { setOpened((current) => current ? annotations.find((item) => item.id === current.id) || null : null); }, [annotations]);
  useEffect(() => { let cancelled = false; const prepareTextLayer = async () => { const textLayer = textRef.current; if (!textLayer || cancelled) return; const viewport = page.getViewport({ scale: 1 }); setPageSize({ width: viewport.width, height: viewport.height }); textLayer.replaceChildren(); textLayer.style.width = `${viewport.width}px`; textLayer.style.height = `${viewport.height}px`; textLayer.style.setProperty("--scale-factor", String(viewport.scale)); textLayer.style.setProperty("--user-unit", String(viewport.userUnit || 1)); textLayer.style.setProperty("--total-scale-factor", String(viewport.scale * (viewport.userUnit || 1))); const content = await page.getTextContent(); if (cancelled) return; pageTextRef.current = content.items.map((item) => item.str).join(" ").replace(/\s+/g, " ").trim(); await new pdfjsLib.TextLayer({ textContentSource: content, container: textLayer, viewport }).render(); }; prepareTextLayer().catch((reason) => { if (!cancelled) onRenderError?.(reason); }); return () => { cancelled = true; }; }, [page]);
  const clearSelection = () => { if (typeof CSS !== "undefined" && CSS.highlights) CSS.highlights.delete(highlightName); window.getSelection?.()?.removeAllRanges(); selectionRectsRef.current = []; setSelectionAction(null); if (activeSelectionPage === page.pageNumber) onSelectionActive(null); };
  const selectedText = (event) => { event.stopPropagation(); const browserSelection = window.getSelection(); const selection = browserSelection?.toString().replace(/\s+/g, " ").trim() || ""; if (selection.length < 2 || !browserSelection?.rangeCount) return; const range = browserSelection.getRangeAt(0).cloneRange(); if (!textRef.current?.contains(range.commonAncestorContainer)) return; const pageRect = pageRef.current?.getBoundingClientRect(); if (!pageRect) return; selectionJustCreatedRef.current = true; selectionRectsRef.current = [...range.getClientRects()].map((rect) => ({ left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom })); if (typeof CSS !== "undefined" && CSS.highlights && typeof Highlight !== "undefined") { CSS.highlights.set(highlightName, new Highlight(range)); browserSelection.removeAllRanges(); } const full = pageTextRef.current; const index = full.toLowerCase().indexOf(selection.toLowerCase()); onSelectionActive(page.pageNumber); setSelectionAction({ text: selection, x: Math.max(2, Math.min(88, ((event.clientX - pageRect.left) / pageRect.width) * 100)), y: Math.max(2, Math.min(90, ((event.clientY - pageRect.top) / pageRect.height) * 100)) }); onTextSelected({ selection, context_before: index < 0 ? "" : full.slice(Math.max(0, index - 900), index), context_after: index < 0 ? "" : full.slice(index + selection.length, index + selection.length + 900) }); };
  const place = (event) => { if (selectionJustCreatedRef.current) { selectionJustCreatedRef.current = false; return; } if (adding && event.target === rasterRef.current) { const rect = rasterRef.current.getBoundingClientRect(); onPlace(page.pageNumber, (event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height); return; } const insideHighlight = selectionRectsRef.current.some((rect) => event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom); if (!insideHighlight && !event.target.closest(".selection-ask-button")) clearSelection(); };
  const moveMarker = (item, position) => { setOpened((current) => current?.id === item.id ? { ...current, page_x: position.x, page_y: position.y } : current); onMove(item, position); };
  const onEdit = (item) => onMove(item, { x: item.page_x, y: item.page_y, note: item.note });
  const scaledSize = pageSize ? { width: `${pageSize.width * zoom}px`, height: `${pageSize.height * zoom}px` } : undefined;
  return <div className="pdf-page" ref={pageRef} onClick={place} style={scaledSize}><div className="pdf-page-content" style={pageSize ? { width: `${pageSize.width}px`, height: `${pageSize.height}px`, transform: `scale(${zoom})` } : undefined}><img ref={rasterRef} className="pdf-page-raster" src={rasterUrl} alt={`PDF 第 ${page.pageNumber} 页`} draggable="false" loading={page.pageNumber === 1 ? "eager" : "lazy"} onError={() => onRenderError?.(new Error("PDF 页面图像加载失败"))} /><div className={`pdf-text-layer textLayer ${adding ? "pdf-text-layer-disabled" : ""}`} ref={textRef} onMouseUp={selectedText} />{selectionAction && activeSelectionPage === page.pageNumber && <button type="button" className="selection-ask-button" style={{ left: `${selectionAction.x}%`, top: `${selectionAction.y}%` }} onClick={(event) => { event.stopPropagation(); onAskSelection(selectionAction.text); }}><MessageSquare size={13} />提问</button>}{annotations.map((item) => <Marker key={item.id} item={item} pageRef={pageRef} onMove={moveMarker} onOpen={setOpened} />)}{draft && <div className="annotation-compose" style={{ left: `${draft.page_x * 100}%`, top: `${draft.page_y * 100}%` }}><textarea autoFocus value={draft.note} onChange={(event) => onSaveDraft({ ...draft, note: event.target.value })} placeholder="写下批注..." /><div><button onClick={() => onSaveDraft(draft, true)}>保存</button><button onClick={onCancelDraft}>取消</button></div></div>}{opened && <aside className="annotation-detail" style={{ left: `${opened.page_x * 100}%`, top: `${opened.page_y * 100}%` }} onClick={(event) => event.stopPropagation()}><button className="annotation-detail-close" onClick={() => setOpened(null)}><X size={14} /></button><textarea className="annotation-edit" value={opened.note} onChange={(event) => setOpened((current) => ({ ...current, note: event.target.value }))} /><button className="annotation-save" onClick={() => onEdit(opened)}>保存</button><button className="annotation-delete" onClick={() => { onDelete(opened.id); setOpened(null); }}><Trash2 size={13} />删除</button></aside>}</div></div>;
}

function PdfView(props) {
  const [pages, setPages] = useState([]);
  const [error, setError] = useState("");
  const [pageErrors, setPageErrors] = useState({});
  const [activeSelectionPage, setActiveSelectionPage] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setPages([]);
    setError("");
    setPageErrors({});

    fetch(props.url)
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const contentType = response.headers.get("content-type") || "";
        if (!contentType.includes("pdf")) throw new Error("服务返回的不是 PDF 文件");
        const data = new Uint8Array(await response.arrayBuffer());

        // Chrome 的 ImageDecoder 会让部分含嵌入图片的论文页只留下空白画布。
        return pdfjsLib.getDocument({
          data,
          isImageDecoderSupported: false,
          isOffscreenCanvasSupported: false,
        }).promise;
      })
      .then(async (pdf) => {
        const loaded = [];
        for (let index = 1; index <= pdf.numPages; index += 1) {
          loaded.push(await pdf.getPage(index));
        }
        if (!cancelled) setPages(loaded);
      })
      .catch((reason) => !cancelled && setError(reason.message));

    return () => { cancelled = true; };
  }, [props.url]);

  if (error) return <div className="pdf-empty">PDF 加载失败：{error}</div>;

  return <div className="pdf-scroll">
    {pages.map((page) => pageErrors[page.pageNumber]
      ? <div className="pdf-empty pdf-page-error" key={page.pageNumber}><FileText size={20} /><p>第 {page.pageNumber} 页渲染失败</p><small>{pageErrors[page.pageNumber]}</small></div>
      : <PdfPage key={page.pageNumber} {...props} page={page} rasterUrl={props.getPageImageUrl(page.pageNumber)} activeSelectionPage={activeSelectionPage} onSelectionActive={setActiveSelectionPage} onRenderError={(reason) => setPageErrors((current) => ({ ...current, [page.pageNumber]: reason?.message || "PDF.js 渲染异常" }))} annotations={props.annotations.filter((item) => item.page_number === page.pageNumber)} />)}
    {!pages.length && <div className="pdf-loading"><LoaderCircle size={20} />正在加载 PDF</div>}
  </div>;
}

export function Reader({ document, state, setState, onBack, onAskSelection }) {
  const [annotations, setAnnotations] = useState([]); const [draft, setDraft] = useState(null); const [adding, setAdding] = useState(false); const [pdfZoom, setPdfZoom] = useState(1.25); const [textZoom, setTextZoom] = useState(1); const [translation, setTranslation] = useState(null); const [loading, setLoading] = useState(false); const [message, setMessage] = useState(""); const translationRequestRef = useRef(0);
  useEffect(() => { api.annotations(state.config.apiUrl, document.id).then((result) => setAnnotations(result.annotations || [])).catch(() => setMessage("批注加载失败")); }, [document.id, state.config.apiUrl]);
  const language = document.metadata?.document_language === "zh" ? "en" : "zh";
  const translate = async (payload) => { const requestId = ++translationRequestRef.current; setLoading(true); setTranslation({ source_text: payload.selection, loading: true }); try { const result = await api.translateSelection(state.config.apiUrl, document.id, language, payload); if (requestId === translationRequestRef.current) setTranslation(result); } catch (error) { if (requestId === translationRequestRef.current) { setTranslation(null); setMessage(`翻译失败：${error.message}`); } } finally { if (requestId === translationRequestRef.current) setLoading(false); } };
  const save = async (value, submit = false) => { setDraft(value); if (!submit || !value.note.trim()) return; try { const item = await api.addAnnotation(state.config.apiUrl, document.id, value); setAnnotations((items) => [...items, item]); setDraft(null); } catch { setMessage("批注保存失败"); } };
  const move = async (item, position) => { const next = { ...item, page_x: position.x, page_y: position.y, note: position.note ?? item.note }; setAnnotations((items) => items.map((entry) => entry.id === item.id ? next : entry)); try { const updated = await api.updateAnnotation(state.config.apiUrl, document.id, item.id, { page_number: next.page_number, page_x: next.page_x, page_y: next.page_y, note: next.note }); setAnnotations((items) => items.map((entry) => entry.id === item.id ? updated : entry)); } catch { setMessage("批注保存失败"); } };
  const edit = async (item) => { if (!item.note.trim()) return; try { const updated = await api.updateAnnotation(state.config.apiUrl, document.id, item.id, { page_number: item.page_number, page_x: item.page_x, page_y: item.page_y, note: item.note }); setAnnotations((items) => items.map((entry) => entry.id === item.id ? updated : entry)); } catch { setMessage("批注保存失败"); } };
  const remove = async (id) => { try { await api.deleteAnnotation(state.config.apiUrl, document.id, id); setAnnotations((items) => items.filter((item) => item.id !== id)); } catch { setMessage("批注删除失败"); } };
  const zoomControls = (value, setter, reset) => <div className="zoom-controls"><button title="缩小" onClick={() => setter((current) => Math.max(.6, current - .15))}><Minus size={13} /></button><output>{Math.round(value * 100)}%</output><button title="放大" onClick={() => setter((current) => Math.min(2.4, current + .15))}><Plus size={13} /></button><button title="重置" onClick={() => setter(reset)}><RotateCcw size={12} /></button></div>;
  return <main className="reader-page">
    <header className="reader-header">
      <div className="reader-title"><span className="eyebrow">论文阅读工作区</span><h2>{document.title}</h2></div>
      <div className="reader-toolbar"><button className={adding ? "reader-view-toggle active" : "reader-view-toggle"} onClick={() => setAdding((value) => !value)}>添加批注</button><button className="reader-back" onClick={onBack}>返回资料库</button></div>
    </header>
    <div className="reader-layout with-pdf with-translation">
      <section className="pdf-pane">
        <div className="pane-heading"><div><strong>原始 PDF</strong><small>{adding ? "点击页面放置批注" : "选中原文即可翻译；高亮会保留至下次选择"}</small></div>{zoomControls(pdfZoom, setPdfZoom, 1.25)}</div>
        <PdfView
          url={api.documentPdfUrl(state.config.apiUrl, document.id)}
          getPageImageUrl={(pageNumber) => api.documentPdfPageImageUrl(state.config.apiUrl, document.id, pageNumber)}
          annotations={annotations}
          zoom={pdfZoom}
          adding={adding}
          draft={draft}
          onPlace={(page, x, y) => { setAdding(false); setDraft({ page_number: page, page_x: x, page_y: y, note: "", color: "#e0b84c" }); }}
          onMove={move}
          onSaveDraft={save}
          onCancelDraft={() => setDraft(null)}
          onTextSelected={translate}
          onDelete={remove}
          onAskSelection={onAskSelection}
        />
      </section>
      <section className="translation-pane">
        <div className="pane-heading"><div><strong>选区译文</strong><small>{loading ? "正在结合全文术语翻译" : translation?.cached ? "已命中本论文缓存" : "选择 PDF 中的原文开始阅读"}</small></div>{zoomControls(textZoom, setTextZoom, 1)}</div>
        {message && <p className="notice">{message}</p>}
        {translation ? <article className="selection-translation" style={{ fontSize: `${13 * textZoom}px` }}><button className="selection-translation-close" onClick={() => setTranslation(null)}><X size={15} /></button><p className="selection-source">{translation.source_text}</p>{translation.loading ? <div className="selection-loading"><LoaderCircle size={18} />正在建立术语上下文</div> : <p className="selection-result">{translation.translated_text}</p>}<small>{translation.cached ? "缓存译文" : "全文术语上下文译文"}</small></article> : <div className="translation-empty"><BookOpen size={25} /><p>在左侧 PDF 中选中一句或一段英文，右侧会显示准确的上下文译文。</p></div>}
      </section>
    </div>
  </main>;
}
