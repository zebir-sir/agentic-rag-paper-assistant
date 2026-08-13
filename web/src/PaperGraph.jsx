import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { BrainCircuit, Focus, Network, Search, X } from "lucide-react";

const CLUSTER_COLORS = [0x69e5ff, 0xbb91ff, 0x72f0c0, 0xffb2d2, 0xffd37a];
const RELATION_LABELS = {
  semantic_similarity: "语义相近",
  cites: "引用",
  method_lineage: "方法演进",
};
const RELATION_COLORS = {
  semantic_similarity: 0x5aa9dc,
  cites: 0xffc365,
  method_lineage: 0x8ce3b0,
};

function createLabelSprite(label, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 512; canvas.height = 96;
  const context = canvas.getContext("2d");
  context.font = "600 36px Inter, Arial, sans-serif";
  context.textAlign = "center";
  context.fillStyle = "rgba(241, 249, 255, 0.96)";
  context.shadowColor = `#${color.toString(16).padStart(6, "0")}`;
  context.shadowBlur = 12;
  context.fillText(label, 256, 56);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(2.5, 0.47, 1);
  return sprite;
}

function createGlowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256; canvas.height = 256;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(128, 128, 2, 128, 128, 128);
  gradient.addColorStop(0, "rgba(221, 250, 255, 0.95)");
  gradient.addColorStop(0.12, "rgba(104, 224, 255, 0.52)");
  gradient.addColorStop(0.42, "rgba(55, 142, 255, 0.16)");
  gradient.addColorStop(1, "rgba(15, 35, 90, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(canvas);
}

function getNodePosition(index, total) {
  const safeTotal = Math.max(total, 1);
  const phi = Math.acos(1 - (2 * (index + 0.5)) / safeTotal);
  const theta = Math.PI * (1 + Math.sqrt(5)) * index;
  const radius = 6.2 + (index % 5) * 0.52;
  return new THREE.Vector3(
    radius * Math.cos(theta) * Math.sin(phi),
    radius * Math.cos(phi) * 1.18,
    radius * Math.sin(theta) * Math.sin(phi),
  );
}

export default function PaperGraph({ graph, onAsk, onClose }) {
  const mountRef = useRef(null);
  const selectedIdRef = useRef(null);
  const selectedEdgeRef = useRef(null);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [focusPoint, setFocusPoint] = useState(null);
  const [query, setQuery] = useState("");
  const selected = graph.nodes.find((node) => node.document_id === selectedId) || null;
  const selectedEdgeSource = graph.nodes.find((node) => node.document_id === selectedEdge?.source_document_id) || null;
  const selectedEdgeTarget = graph.nodes.find((node) => node.document_id === selectedEdge?.target_document_id) || null;

  useEffect(() => { selectedIdRef.current = selectedId; }, [selectedId]);
  useEffect(() => { selectedEdgeRef.current = selectedEdge; }, [selectedEdge]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x020612, 0.045);
    const camera = new THREE.PerspectiveCamera(52, mount.clientWidth / mount.clientHeight, 0.1, 100);
    camera.position.set(0, 0, 15);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);
    const root = new THREE.Group(); scene.add(root);
    scene.add(new THREE.AmbientLight(0x647dba, 0.8));
    const pointLight = new THREE.PointLight(0x8cecff, 24, 32); pointLight.position.set(0, 1, 7); scene.add(pointLight);
    const glowTexture = createGlowTexture();
    const core = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.85 }));
    core.scale.set(7, 7, 1); root.add(core);
    const coreRing = new THREE.Mesh(new THREE.TorusGeometry(2.1, 0.012, 8, 96), new THREE.MeshBasicMaterial({ color: 0x5bdcff, transparent: true, opacity: 0.28, blending: THREE.AdditiveBlending }));
    coreRing.rotation.x = Math.PI / 2.15; root.add(coreRing);
    const starGeometry = new THREE.BufferGeometry();
    const stars = new Float32Array(1500 * 3);
    for (let index = 0; index < stars.length; index += 3) { stars[index] = (Math.random() - 0.5) * 46; stars[index + 1] = (Math.random() - 0.5) * 30; stars[index + 2] = (Math.random() - 0.5) * 30; }
    starGeometry.setAttribute("position", new THREE.BufferAttribute(stars, 3));
    const starMaterial = new THREE.PointsMaterial({ color: 0x9fc6e7, size: 0.035, transparent: true, opacity: 0.74, blending: THREE.AdditiveBlending });
    scene.add(new THREE.Points(starGeometry, starMaterial));
    const dustGeometry = new THREE.BufferGeometry();
    const dust = new Float32Array(1300 * 3);
    for (let index = 0; index < dust.length; index += 3) {
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.pow(Math.random(), 0.55) * 9;
      dust[index] = Math.cos(angle) * radius;
      dust[index + 1] = (Math.random() - 0.5) * (0.7 + radius * 0.12);
      dust[index + 2] = Math.sin(angle) * radius * 0.62;
    }
    dustGeometry.setAttribute("position", new THREE.BufferAttribute(dust, 3));
    const dustCloud = new THREE.Points(dustGeometry, new THREE.PointsMaterial({ color: 0x4dc7ff, size: 0.055, transparent: true, opacity: 0.23, blending: THREE.AdditiveBlending }));
    root.add(dustCloud);
    const nodeMeshes = [];
    const edgeMeshes = [];
    const nodeLabels = [];
    const nodeHalos = [];
    const positions = new Map();
    graph.nodes.forEach((node, index) => {
      const color = CLUSTER_COLORS[index % CLUSTER_COLORS.length];
      const position = getNodePosition(index, graph.nodes.length); positions.set(node.document_id, position);
      const material = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.7, roughness: 0.24, metalness: 0.18 });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.22 + Math.min(node.chunk_count || 0, 30) / 170, 28, 28), material);
      mesh.position.copy(position); mesh.userData.node = node; mesh.userData.baseScale = 1; root.add(mesh); nodeMeshes.push(mesh);
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture, color, transparent: true, opacity: 0.34, blending: THREE.AdditiveBlending, depthWrite: false }));
      halo.position.copy(position); halo.scale.set(1.5, 1.5, 1); halo.userData.documentId = node.document_id; root.add(halo); nodeHalos.push(halo);
      const label = createLabelSprite(node.abbreviation, color); label.position.copy(position).add(new THREE.Vector3(0, 0.48, 0)); label.userData.documentId = node.document_id; root.add(label); nodeLabels.push(label);
    });
    graph.edges.forEach((edge) => {
      const source = positions.get(edge.source_document_id); const target = positions.get(edge.target_document_id);
      if (!source || !target) return;
      const geometry = new THREE.BufferGeometry().setFromPoints([source, target]);
      const relationColor = RELATION_COLORS[edge.relation_type] || RELATION_COLORS.semantic_similarity;
      const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: relationColor, transparent: true, opacity: Math.min(0.7, 0.12 + edge.score * 0.5) }));
      line.userData.edge = edge; line.userData.baseOpacity = line.material.opacity;
      root.add(line); edgeMeshes.push(line);
    });
    const raycaster = new THREE.Raycaster(); const pointer = new THREE.Vector2();
    raycaster.params.Line.threshold = 0.18;
    let dragging = false; let moved = false; let lastX = 0; let lastY = 0;
    const selectGraphItem = (event) => {
      if (moved) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1; pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const nodeHit = raycaster.intersectObjects(nodeMeshes)[0];
      if (nodeHit) {
        const nodeId = nodeHit.object.userData.node.document_id;
        selectedEdgeRef.current = null; selectedIdRef.current = nodeId;
        setFocusPoint({ x: event.clientX - bounds.left, y: event.clientY - bounds.top });
        setSelectedEdge(null); setSelectedId(nodeId); return;
      }
      const edgeHit = raycaster.intersectObjects(edgeMeshes)[0];
      if (edgeHit) {
        const edge = edgeHit.object.userData.edge;
        selectedIdRef.current = null; selectedEdgeRef.current = edge;
        setFocusPoint({ x: event.clientX - bounds.left, y: event.clientY - bounds.top });
        setSelectedId(null); setSelectedEdge(edge); return;
      }
      selectedIdRef.current = null; selectedEdgeRef.current = null;
      setFocusPoint(null); setSelectedId(null); setSelectedEdge(null);
    };
    renderer.domElement.addEventListener("click", selectGraphItem);
    const down = (event) => { if (selectedIdRef.current || selectedEdgeRef.current) return; dragging = true; moved = false; lastX = event.clientX; lastY = event.clientY; };
    const move = (event) => { if (!dragging) return; const dx = event.clientX - lastX; const dy = event.clientY - lastY; if (Math.abs(dx) + Math.abs(dy) > 2) moved = true; root.rotation.y += dx * 0.008; root.rotation.x += dy * 0.008; lastX = event.clientX; lastY = event.clientY; };
    const up = () => { dragging = false; };
    renderer.domElement.addEventListener("pointerdown", down); window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    const resize = () => { camera.aspect = mount.clientWidth / mount.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(mount.clientWidth, mount.clientHeight); };
    const observer = new ResizeObserver(resize); observer.observe(mount);
    let frame;
    const render = () => {
      const focusedNodeId = selectedIdRef.current;
      const focusedEdge = selectedEdgeRef.current;
      const focusActive = Boolean(focusedNodeId || focusedEdge);
      const highlightedNodeIds = focusedEdge ? new Set([focusedEdge.source_document_id, focusedEdge.target_document_id]) : new Set(focusedNodeId ? [focusedNodeId] : []);
      starMaterial.opacity = focusActive ? 0.18 : 0.72;
      nodeMeshes.forEach((mesh) => {
        const highlighted = highlightedNodeIds.has(mesh.userData.node.document_id);
        const material = mesh.material;
        material.transparent = focusActive;
        material.opacity = focusActive && !highlighted ? 0.14 : 1;
        material.emissiveIntensity = highlighted ? 3.4 : focusActive ? 0.22 : 1.7;
        const targetScale = highlighted ? 1.75 : 1;
        mesh.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.16);
      });
      nodeLabels.forEach((label) => {
        const highlighted = highlightedNodeIds.has(label.userData.documentId);
        label.material.opacity = focusActive && !highlighted ? 0.16 : highlighted ? 1 : 0.96;
        const targetScale = highlighted ? 1.28 : 1;
        label.scale.lerp(new THREE.Vector3(2.5 * targetScale, 0.47 * targetScale, 1), 0.16);
      });
      nodeHalos.forEach((halo) => {
        const highlighted = highlightedNodeIds.has(halo.userData.documentId);
        halo.material.opacity = focusActive && !highlighted ? 0.035 : highlighted ? 0.82 : 0.34;
        const targetScale = highlighted ? 2.25 : 1.5;
        halo.scale.lerp(new THREE.Vector3(targetScale, targetScale, 1), 0.16);
      });
      edgeMeshes.forEach((line) => {
        const highlighted = focusedEdge && line.userData.edge === focusedEdge;
        const adjacent = focusedNodeId && (line.userData.edge.source_document_id === focusedNodeId || line.userData.edge.target_document_id === focusedNodeId);
        line.material.opacity = highlighted ? 1 : focusActive ? (adjacent ? 0.72 : 0.055) : line.userData.baseOpacity;
        line.material.linewidth = highlighted ? 3 : 1;
      });
      core.material.opacity = 0.68 + Math.sin(performance.now() * 0.0014) * 0.14;
      coreRing.rotation.z += 0.0016;
      dustCloud.rotation.y += 0.00045;
      if (!focusActive && !dragging) root.rotation.y += 0.0012;
      renderer.render(scene, camera); frame = requestAnimationFrame(render);
    };
    render();
    return () => { cancelAnimationFrame(frame); observer.disconnect(); renderer.domElement.removeEventListener("click", selectGraphItem); renderer.domElement.removeEventListener("pointerdown", down); window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); renderer.dispose(); mount.removeChild(renderer.domElement); };
  }, [graph]);

  const focus = () => { const matched = graph.nodes.find((node) => `${node.title} ${node.title_zh} ${node.abbreviation}`.toLowerCase().includes(query.toLowerCase())); if (matched) { setFocusPoint({ x: 48, y: 108 }); setSelectedEdge(null); setSelectedId(matched.document_id); } };
  const card = selected?.research_card || {};
  const detailStyle = focusPoint ? { left: `clamp(18px, ${focusPoint.x + 18}px, calc(100% - 373px))`, top: `clamp(18px, ${focusPoint.y + 18}px, calc(100% - 390px))` } : undefined;
  const clearFocus = () => { selectedIdRef.current = null; selectedEdgeRef.current = null; setFocusPoint(null); setSelectedId(null); setSelectedEdge(null); };
  const openEdgeEndpoint = (documentId) => { selectedEdgeRef.current = null; selectedIdRef.current = documentId; setSelectedEdge(null); setSelectedId(documentId); };
  return <main className="paper-graph-page"><header className="graph-header"><div><span className="eyebrow">论文知识星图</span><h2>研究关系网络</h2><p>关系用于扩展检索范围；最终回答仍回到正文、图表和算法证据。</p></div><button className="reader-back" onClick={onClose}><X size={15} />返回工作台</button></header><section className={`graph-canvas-shell ${selected || selectedEdge ? "graph-focused" : ""}`}><div className="graph-atmosphere" aria-hidden="true"><i /><i /><i /></div><div className="graph-toolbar"><label><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === "Enter" && focus()} placeholder="中文标题、原题或缩写" /></label><button onClick={focus} title="定位论文"><Focus size={16} /></button><span><Network size={14} />{graph.nodes.length} 篇论文 · {graph.edges.length} 条关系</span></div><div className="graph-legend" aria-label="关系类型图例"><span className="relation-semantic_similarity">语义相近</span><span className="relation-cites">引用</span><span className="relation-method_lineage">方法演进</span></div><div className="graph-canvas" ref={mountRef} />{selected && <aside className="graph-detail graph-detail-anchored" style={detailStyle}><button className="graph-detail-close" onClick={clearFocus} title="关闭详情"><X size={16} /></button><span className="eyebrow">{selected.abbreviation}</span><h3>{selected.title_zh || "正在生成中文研究卡片"}</h3><p className="graph-original-title">{selected.title}</p>{selected.localization_status === "ready" ? <><div className="graph-keywords">{(card.keywords_zh || []).map((keyword) => <span key={keyword}>{keyword}</span>)}</div><dl className="graph-card"><dt>研究问题</dt><dd>{card.problem_zh}</dd><dt>核心方法</dt><dd>{card.method_zh}</dd><dt>创新切入</dt><dd>{card.innovation_zh}</dd></dl></> : <p className="graph-localization-status">{selected.localization_status === "failed" ? "中文卡片暂不可用，保留原始论文信息。" : "正在从原论文生成并校验中文研究卡片..."}</p>}<p className="graph-relation-hint">点击星图中的连线查看具体关系与原文依据。</p><button className="primary-button graph-ask" onClick={() => onAsk(selected)}><BrainCircuit size={16} />围绕此论文探索创新</button></aside>}{selectedEdge && <aside className="graph-detail graph-edge-detail graph-detail-anchored" style={detailStyle}><button className="graph-detail-close" onClick={clearFocus} title="关闭关系详情"><X size={16} /></button><span className="eyebrow">论文关系</span><h3>{RELATION_LABELS[selectedEdge.relation_type] || selectedEdge.relation_type}</h3><div className="graph-edge-papers"><button onClick={() => openEdgeEndpoint(selectedEdgeSource?.document_id || null)}>{selectedEdgeSource?.abbreviation || "来源论文"}</button><span>→</span><button onClick={() => openEdgeEndpoint(selectedEdgeTarget?.document_id || null)}>{selectedEdgeTarget?.abbreviation || "目标论文"}</button></div><dl className="graph-card graph-edge-card"><dt>关系强度</dt><dd>{Math.round(selectedEdge.score * 100)}%</dd><dt>关系依据</dt><dd>{selectedEdge.evidence?.explanation || "该关系由论文内容相似度或可追溯的正文证据建立。"}</dd>{selectedEdge.evidence?.source_section && <><dt>证据章节</dt><dd>{selectedEdge.evidence.source_section}</dd></>}</dl><p className="graph-relation-hint">点击两端缩写可查看对应论文卡片。</p></aside>}</section></main>;
}
