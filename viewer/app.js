// viewer/app.js

// 🔹 Path to results (works locally + GitHub Pages)
const RESULTS_PATH = "/results/latest.json";

// 🔹 DOM elements
const metadataEl = document.getElementById("metadata-content");
const promptEl = document.getElementById("prompt-content");
const modelsContainer = document.getElementById("models-container");
const finalOutputEl = document.getElementById("final-output");
const errorSection = document.getElementById("error-section");
const errorMessage = document.getElementById("error-message");

// 🔹 Utility: show error
function showError(message) {
  errorSection.classList.remove("hidden");
  errorMessage.textContent = message;
}

// 🔹 Utility: format text (fix \n issue)
function formatText(text) {
  if (!text) return "<em>No content</em>";

  return text
    .replace(/\n/g, "<br>")  // preserve line breaks
    .replace(/\t/g, "&nbsp;&nbsp;&nbsp;&nbsp;");
}

// 🔹 Utility: safely get nested values
function safeGet(obj, path, fallback = "N/A") {
  try {
    return path.split(".").reduce((acc, key) => acc[key], obj) ?? fallback;
  } catch {
    return fallback;
  }
}

// 🔹 Render metadata
function renderMetadata(data) {
  const timestamp = data.timestamp || "N/A";
  const architecture = data.architecture || "N/A";

  metadataEl.innerHTML = `
    <p><strong>Timestamp:</strong> ${timestamp}</p>
    <p><strong>Architecture:</strong> ${architecture}</p>
  `;
}

// 🔹 Render prompt
function renderPrompt(data) {
  const prompt =
    safeGet(data, "cases.0.prompt") ||
    safeGet(data, "prompt") ||
    "Prompt not found";

  promptEl.innerHTML = formatText(prompt);
}

// 🔹 Create collapsible model section
function createModelBlock(title, contentObj) {
  const wrapper = document.createElement("div");
  wrapper.className = "model-block";

  const header = document.createElement("div");
  header.className = "model-header";
  header.textContent = title;

  const body = document.createElement("div");
  body.className = "model-body hidden";

  const hallucinations = contentObj.hallucinations_found ?? "N/A";
  const types = (contentObj.types || []).join(", ") || "None";
  const justification = contentObj.justification || "N/A";
  const corrected = contentObj.corrected_answer || contentObj.output || "N/A";

  body.innerHTML = `
    <p><strong>Hallucinations Found:</strong> ${hallucinations}</p>
    <p><strong>Types:</strong> ${types}</p>
    <p><strong>Justification:</strong><br>${formatText(justification)}</p>
    <p><strong>Output:</strong><br>${formatText(corrected)}</p>
  `;

  // Toggle behavior
  header.onclick = () => {
    body.classList.toggle("hidden");
  };

  wrapper.appendChild(header);
  wrapper.appendChild(body);

  return wrapper;
}

// 🔹 Render model outputs
function renderModels(data) {
  modelsContainer.innerHTML = "";

  const caseData = data.cases?.[0];
  if (!caseData) {
    modelsContainer.innerHTML = "<p>No model outputs found.</p>";
    return;
  }

  // Try to support both architectures
  const modelOutputs =
    caseData.model_outputs ||
    caseData.outputs ||
    caseData.steps ||
    [];

  if (!Array.isArray(modelOutputs) || modelOutputs.length === 0) {
    modelsContainer.innerHTML = "<p>No model outputs available.</p>";
    return;
  }

  modelOutputs.forEach((model, index) => {
    const title =
      model.role
        ? `${model.role} (${model.model || "model"})`
        : `Model ${index + 1}`;

    const block = createModelBlock(title, model.output || model);

    modelsContainer.appendChild(block);
  });
}

// 🔹 Render final output
function renderFinalOutput(data) {
  const caseData = data.cases?.[0];

  const final =
    caseData?.final_output ||
    caseData?.output ||
    data.final_output ||
    "Final output not found";

  finalOutputEl.innerHTML = formatText(final);
}

// 🔹 Main init
async function init() {
  try {
    const response = await fetch(RESULTS_PATH);

    if (!response.ok) {
      throw new Error(`Failed to load JSON (${response.status})`);
    }

    const data = await response.json();

    renderMetadata(data);
    renderPrompt(data);
    renderModels(data);
    renderFinalOutput(data);

  } catch (err) {
    console.error(err);
    showError("Failed to load results. Check path or GitHub Pages setup.");
  }
}

// 🔹 Run
init();
