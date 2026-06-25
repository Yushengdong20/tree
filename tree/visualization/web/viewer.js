const treeRoot = document.getElementById("tree-root");
const executionState = document.getElementById("execution-state");
const rootStatus = document.getElementById("root-status");
const nodeCount = document.getElementById("node-count");
const tickCount = document.getElementById("tick-count");
const tickInterval = document.getElementById("tick-interval");
const liveRuntime = document.getElementById("live-runtime");
const template = document.getElementById("node-template");
const refreshBtn = document.getElementById("refresh-btn");
const collapsedNodes = new Set();

function statusClass(status) {
  const key = (status || "unknown").toLowerCase();
  if (["success", "failure", "running", "invalid"].includes(key)) {
    return `status-${key}`;
  }
  return "status-unknown";
}

function renderNode(node) {
  const fragment = template.content.firstElementChild.cloneNode(true);
  fragment.querySelector(".node-label").textContent = node.label || node.name;
  fragment.querySelector(".node-type").textContent = node.type || "Node";

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

  if (children.length === 0) {
    cardEl.classList.add("is-leaf");
  } else {
    const nodeId = node.id || "";
    const collapsed = collapsedNodes.has(nodeId);
    toggleBtn.textContent = collapsed ? "+" : "-";
    childrenEl.classList.toggle("is-collapsed", collapsed);
    toggleBtn.addEventListener("click", () => {
      if (collapsedNodes.has(nodeId)) {
        collapsedNodes.delete(nodeId);
      } else {
        collapsedNodes.add(nodeId);
      }
      refresh();
    });
  }

  children.forEach((child) => {
    childrenEl.appendChild(renderNode(child));
  });
  return fragment;
}

function renderTree(snapshot) {
  executionState.textContent = snapshot.execution_state || "UNKNOWN";
  executionState.className = statusClass(snapshot.execution_state);
  rootStatus.textContent = snapshot.root_status || "UNKNOWN";
  rootStatus.className = statusClass(snapshot.root_status);
  nodeCount.textContent = String(snapshot.node_count || 0);
  tickCount.textContent = String(snapshot.tick_count || 0);
  tickInterval.textContent = snapshot.last_tick_interval
    ? `${(snapshot.last_tick_interval * 1000).toFixed(0)} ms`
    : "-";
  renderLiveRuntime(snapshot.live_runtime);

  treeRoot.innerHTML = "";
  if (!snapshot.tree) {
    treeRoot.textContent = "Waiting for the behaviour tree...";
    return;
  }
  treeRoot.appendChild(renderNode(snapshot.tree));
}

function renderLiveRuntime(live) {
  if (!live) {
    liveRuntime.textContent = "No blocking runtime step is active.";
    liveRuntime.className = "live-runtime empty";
    return;
  }
  liveRuntime.className = "live-runtime";
  liveRuntime.innerHTML = `
    <strong>${live.active_node || "unknown node"}</strong>
    <span>${live.phase || "RUNNING"}</span>
    <div>${live.detail || ""}</div>
  `;
}

async function refresh() {
  try {
    const response = await fetch("./api/state", { cache: "no-store" });
    const snapshot = await response.json();
    renderTree(snapshot);
  } catch (error) {
    treeRoot.textContent = `Failed to load state: ${error.message}`;
  }
}

refreshBtn.addEventListener("click", refresh);
refresh();
setInterval(refresh, 500);
