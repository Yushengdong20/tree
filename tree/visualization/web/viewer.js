const treeRoot = document.getElementById("tree-root");
const executionState = document.getElementById("execution-state");
const rootStatus = document.getElementById("root-status");
const nodeCount = document.getElementById("node-count");
const tickCount = document.getElementById("tick-count");
const tickInterval = document.getElementById("tick-interval");
const template = document.getElementById("node-template");
const refreshBtn = document.getElementById("refresh-btn");
const collapsedNodes = new Set();
const refreshIntervalMs = 100;
const minTreeScale = 0.35;
const maxTreeScale = 1.6;
const treeScaleStep = 0.08;
const nodeWidth = 210;
const nodeHeight = 178;
const siblingGap = 16;
const levelGap = 72;
const canvasPadding = 36;
let isRefreshing = false;
let treeScale = 0.82;
let treeContent = null;
let lastFocusedNodeId = "";
let isDraggingTree = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartScrollLeft = 0;
let dragStartScrollTop = 0;

function statusClass(status) {
  const key = (status || "unknown").toLowerCase();
  if (["success", "failure", "running", "invalid"].includes(key)) {
    return `status-${key}`;
  }
  return "status-unknown";
}

function visibleChildren(node) {
  const nodeId = node.id || "";
  if (collapsedNodes.has(nodeId)) {
    return [];
  }
  return node.children || [];
}

function buildLayout(node, depth = 0) {
  const children = visibleChildren(node).map((child) => buildLayout(child, depth + 1));
  const leftContour = new Map();
  const rightContour = new Map();

  for (const child of children) {
    // 按每一层的真实轮廓压紧相邻子树，避免深层分支把整层硬撑宽。
    const shiftX = calculateContourShift(rightContour, child.leftContour);
    shiftLayout(child, shiftX);
    mergeContours(leftContour, rightContour, child.leftContour, child.rightContour);
  }

  const x = children.length > 0
    ? (children[0].x + children[children.length - 1].x) / 2
    : 0;
  leftContour.set(depth, Math.min(leftContour.get(depth) ?? Infinity, x - nodeWidth / 2));
  rightContour.set(depth, Math.max(rightContour.get(depth) ?? -Infinity, x + nodeWidth / 2));

  const layout = {
    node,
    children,
    depth,
    leftContour,
    rightContour,
    x,
    y: depth * (nodeHeight + levelGap),
  };
  return layout;
}

function calculateContourShift(rightContour, leftContour) {
  if (rightContour.size === 0) {
    return 0;
  }
  let shiftX = 0;
  for (const [depth, left] of leftContour.entries()) {
    const right = rightContour.get(depth);
    if (typeof right === "number") {
      shiftX = Math.max(shiftX, right + siblingGap - left);
    }
  }
  return shiftX;
}

function shiftLayout(layout, shiftX) {
  if (shiftX === 0) {
    return;
  }
  layout.x += shiftX;
  shiftContour(layout.leftContour, shiftX);
  shiftContour(layout.rightContour, shiftX);
  for (const child of layout.children) {
    shiftLayout(child, shiftX);
  }
}

function shiftContour(contour, shiftX) {
  for (const [depth, value] of contour.entries()) {
    contour.set(depth, value + shiftX);
  }
}

function mergeContours(targetLeft, targetRight, sourceLeft, sourceRight) {
  for (const [depth, left] of sourceLeft.entries()) {
    targetLeft.set(depth, Math.min(targetLeft.get(depth) ?? Infinity, left));
  }
  for (const [depth, right] of sourceRight.entries()) {
    targetRight.set(depth, Math.max(targetRight.get(depth) ?? -Infinity, right));
  }
}

function collectLayoutNodes(layout, output = []) {
  output.push(layout);
  for (const child of layout.children) {
    collectLayoutNodes(child, output);
  }
  return output;
}

function normalizeLayout(layout, shiftX) {
  layout.x += shiftX;
  shiftContour(layout.leftContour, shiftX);
  shiftContour(layout.rightContour, shiftX);
  for (const child of layout.children) {
    normalizeLayout(child, shiftX);
  }
}

function getLayoutBounds(layout) {
  const left = Math.min(...layout.leftContour.values());
  const right = Math.max(...layout.rightContour.values());
  return { left, right };
}

function createNodeElement(node, focusNodeId) {
  const fragment = template.content.firstElementChild.cloneNode(true);
  const nodeId = node.id || "";
  fragment.dataset.nodeId = nodeId;
  fragment.querySelector(".node-label").textContent = node.label || node.name;
  fragment.querySelector(".node-type").textContent = node.type || "Node";
  fragment.querySelector(".node-timing").textContent = formatTiming(node.timing);

  const history = [];
  if (node.last_terminal_status && node.status === "INVALID") {
    history.push(`last terminal: ${node.last_terminal_status}`);
  }
  fragment.querySelector(".node-history").textContent = history.join(" | ");

  const statusEl = fragment.querySelector(".node-status");
  statusEl.textContent = node.status || "Unknown";
  statusEl.className = `node-status ${statusClass(node.status)}`;

  const cardEl = fragment.querySelector(".node-card");
  const toggleBtn = fragment.querySelector(".node-toggle");
  const childrenEl = fragment.querySelector(".node-children");
  const children = node.children || [];
  childrenEl.remove();
  if (node.status === "RUNNING") {
    cardEl.classList.add("is-running");
    cardEl.dataset.nodeId = nodeId;
  }
  if (nodeId === focusNodeId) {
    cardEl.classList.add("is-focus-target");
    if (node.status === "FAILURE") {
      cardEl.classList.add("is-failure-target");
    }
  }

  if (children.length === 0) {
    cardEl.classList.add("is-leaf");
  } else {
    fragment.classList.add("has-children");
    const collapsed = collapsedNodes.has(nodeId);
    fragment.classList.toggle("is-collapsed", collapsed);
    toggleBtn.textContent = collapsed ? "+" : "-";
    toggleBtn.addEventListener("click", () => {
      if (collapsedNodes.has(nodeId)) {
        collapsedNodes.delete(nodeId);
      } else {
        collapsedNodes.add(nodeId);
      }
      refresh();
    });
  }
  return fragment;
}

function createLinkPath(parent, child) {
  const startX = parent.x;
  const startY = parent.y + canvasPadding + nodeHeight;
  const endX = child.x;
  const endY = child.y + canvasPadding;
  const midY = startY + (endY - startY) / 2;
  return `M ${startX} ${startY} V ${midY} H ${endX} V ${endY}`;
}

function appendLayoutLinks(svg, layout) {
  for (const child of layout.children) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "tree-link");
    path.setAttribute("d", createLinkPath(layout, child));
    svg.appendChild(path);
    appendLayoutLinks(svg, child);
  }
}

function renderLayoutTree(tree, focusNodeId) {
  const layout = buildLayout(tree);
  const bounds = getLayoutBounds(layout);
  normalizeLayout(layout, canvasPadding - bounds.left);
  const nodes = collectLayoutNodes(layout);
  const normalizedBounds = getLayoutBounds(layout);
  const maxDepth = nodes.reduce((maxValue, item) => Math.max(maxValue, item.depth), 0);
  const canvasWidth = normalizedBounds.right + canvasPadding;
  const canvasHeight = (maxDepth + 1) * nodeHeight + maxDepth * levelGap + canvasPadding * 2;

  treeContent = document.createElement("div");
  treeContent.className = "tree-content";
  treeContent.style.width = `${canvasWidth}px`;
  treeContent.style.height = `${canvasHeight}px`;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "tree-links");
  svg.setAttribute("width", String(canvasWidth));
  svg.setAttribute("height", String(canvasHeight));
  svg.setAttribute("viewBox", `0 0 ${canvasWidth} ${canvasHeight}`);
  appendLayoutLinks(svg, layout);
  treeContent.appendChild(svg);

  for (const item of nodes) {
    const nodeEl = createNodeElement(item.node, focusNodeId);
    nodeEl.classList.add("layout-node");
    nodeEl.style.left = `${item.x - nodeWidth / 2}px`;
    nodeEl.style.top = `${item.y + canvasPadding}px`;
    treeContent.appendChild(nodeEl);
  }
  return treeContent;
}

function findDeepestRunningNodeId(node) {
  if (!node) {
    return "";
  }
  // 行为树运行时父节点也可能是 RUNNING，优先找更深层的实际执行节点。
  const children = node.children || [];
  for (const child of children) {
    const runningNodeId = findDeepestRunningNodeId(child);
    if (runningNodeId) {
      return runningNodeId;
    }
  }
  if (node.status === "RUNNING") {
    return node.id || "";
  }
  return "";
}

function findDeepestFailureNodeId(node) {
  if (!node) {
    return "";
  }
  // 失败时优先定位更深层的具体失败节点，而不是外层 Sequence/Selector。
  const children = node.children || [];
  for (const child of children) {
    const failureNodeId = findDeepestFailureNodeId(child);
    if (failureNodeId) {
      return failureNodeId;
    }
  }
  if (node.status === "FAILURE") {
    return node.id || "";
  }
  return "";
}

function findFocusNodeId(tree) {
  return findDeepestRunningNodeId(tree) || findDeepestFailureNodeId(tree);
}

function expandAncestors(nodeId) {
  if (!nodeId) {
    return;
  }
  // 自动展开正在执行节点的父链，避免目标节点被用户之前的折叠状态隐藏。
  const parts = nodeId.split("/");
  for (let index = 1; index < parts.length; index += 1) {
    collapsedNodes.delete(parts.slice(0, index).join("/"));
  }
}

function focusTreeNode(focusNodeId) {
  if (!focusNodeId) {
    lastFocusedNodeId = "";
    return;
  }
  if (focusNodeId === lastFocusedNodeId) {
    return;
  }
  lastFocusedNodeId = focusNodeId;
  // 等 DOM 重建后的布局完成，再执行定位，避免高频刷新时滚动目标还没稳定。
  requestAnimationFrame(() => {
    const focusNode = treeRoot.querySelector(".node-card.is-focus-target");
    if (!focusNode) {
      return;
    }
    // 仅在焦点节点变化时定位，避免用户拖动画布时被持续抢回。
    focusNode.scrollIntoView({
      behavior: "auto",
      block: "center",
      inline: "nearest",
    });
  });
}

function clamp(value, minValue, maxValue) {
  return Math.min(maxValue, Math.max(minValue, value));
}

function applyTreeScale() {
  if (!treeContent) {
    return;
  }
  treeContent.style.transform = `scale(${treeScale})`;
}

function handleTreeWheel(event) {
  if (isDraggingTree) {
    return;
  }
  event.preventDefault();
  const direction = event.deltaY > 0 ? -1 : 1;
  treeScale = clamp(
    treeScale + direction * treeScaleStep,
    minTreeScale,
    maxTreeScale
  );
  applyTreeScale();
}

function handleTreePointerDown(event) {
  if (event.button !== 0) {
    return;
  }
  isDraggingTree = true;
  dragStartX = event.clientX;
  dragStartY = event.clientY;
  dragStartScrollLeft = treeRoot.scrollLeft;
  dragStartScrollTop = treeRoot.scrollTop;
  treeRoot.classList.add("is-dragging");
  treeRoot.setPointerCapture(event.pointerId);
}

function handleTreePointerMove(event) {
  if (!isDraggingTree) {
    return;
  }
  event.preventDefault();
  treeRoot.scrollLeft = dragStartScrollLeft - (event.clientX - dragStartX);
  treeRoot.scrollTop = dragStartScrollTop - (event.clientY - dragStartY);
}

function finishTreeDrag(event) {
  if (!isDraggingTree) {
    return;
  }
  isDraggingTree = false;
  treeRoot.classList.remove("is-dragging");
  if (treeRoot.hasPointerCapture(event.pointerId)) {
    treeRoot.releasePointerCapture(event.pointerId);
  }
}

function formatTiming(timing) {
  if (!timing) {
    return "";
  }
  const badge = timing.is_subtree ? "subtree" : (timing.is_leaf ? "leaf" : "node");
  return `${badge} | last ${formatSeconds(timing.last_elapsed_sec)} | avg ${formatSeconds(timing.avg_elapsed_sec)} | total ${formatSeconds(timing.total_elapsed_sec)} | count ${timing.count}`;
}

function formatSeconds(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(3)}s`;
}

function renderTree(snapshot) {
  const focusNodeId = findFocusNodeId(snapshot.tree);
  expandAncestors(focusNodeId);

  executionState.textContent = snapshot.execution_state || "UNKNOWN";
  executionState.className = statusClass(snapshot.execution_state);
  rootStatus.textContent = snapshot.root_status || "UNKNOWN";
  rootStatus.className = statusClass(snapshot.root_status);
  nodeCount.textContent = String(snapshot.node_count || 0);
  tickCount.textContent = String(snapshot.tick_count || 0);
  tickInterval.textContent = snapshot.last_tick_interval
    ? `${(snapshot.last_tick_interval * 1000).toFixed(0)} ms`
    : "-";

  treeRoot.innerHTML = "";
  if (!snapshot.tree) {
    treeRoot.textContent = "Waiting for the behaviour tree...";
    return;
  }
  treeRoot.appendChild(renderLayoutTree(snapshot.tree, focusNodeId));
  applyTreeScale();
  focusTreeNode(focusNodeId);
}

async function refresh() {
  if (isRefreshing) {
    return;
  }
  // 高频轮询时避免上一次请求未完成又发起新的请求。
  isRefreshing = true;
  try {
    const response = await fetch("./api/state", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const snapshot = await response.json();
    renderTree(snapshot);
  } catch (error) {
    treeRoot.textContent = `Failed to load state: ${error.message}`;
  } finally {
    isRefreshing = false;
  }
}

refreshBtn.addEventListener("click", refresh);
treeRoot.addEventListener("wheel", handleTreeWheel, { passive: false });
treeRoot.addEventListener("pointerdown", handleTreePointerDown);
treeRoot.addEventListener("pointermove", handleTreePointerMove);
treeRoot.addEventListener("pointerup", finishTreeDrag);
treeRoot.addEventListener("pointercancel", finishTreeDrag);
refresh();
setInterval(refresh, refreshIntervalMs);
