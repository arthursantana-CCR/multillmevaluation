import fs from "fs/promises";
import path from "path";
import process from "process";
import YAML from "yaml";
import { buildMarkdown } from "./render_results.mjs";

const CONFIG_PATH = path.resolve("eval_config.yaml");
const RESULTS_DIR = path.resolve("results");
const HISTORY_DIR = path.resolve("results/history");

// 🔹 NEW
const RESULTS_MD_DIR = path.resolve("results_md");
const HISTORY_MD_DIR = path.resolve("results_md/history");

const MAX_RETRIES = 1;

// ================== HELPERS ==================

function buildModelSequence(sequence) {
  return sequence.map((m) => `${m.model} (${m.role || "candidate"})`);
}

function isModelError(text) {
  return typeof text === "string" && text.startsWith("[ERROR:");
}

function detectErrorType(text) {
  if (!text || typeof text !== "string") return "unknown_error";
  if (text.toLowerCase().includes("quota")) return "quota_exceeded";
  if (text.includes("429")) return "rate_limit_or_quota";
  if (text.includes("503")) return "service_unavailable";
  if (text.toLowerCase().includes("blocked")) return "prompt_blocked";
  if (text.toLowerCase().includes("no candidates")) return "no_candidates";
  if (text.toLowerCase().includes("empty text")) return "empty_text";
  if (text.toLowerCase().includes("malformed")) return "malformed_response";
  return "provider_error";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status) {
  return status === 429 || status === 500 || status === 503 || status === 504;
}

// 🔹 NEW: unified task handler
function buildTaskInput({ config, caseConfig }) {
  if (config.task_type === "generation") {
    return config.task;
  }

  if (config.task_type === "evaluation") {
    return `${config.task}\n\n${JSON.stringify(caseConfig)}`;
  }

  throw new Error(`Unknown task_type: ${config.task_type}`);
}

function buildFallbackObject(rawText, fallbackText = "") {
  return {
    hallucinations_found: false,
    types: [],
    justification: "Fallback: unable to parse model output.",
    corrected_answer:
      typeof fallbackText === "string" && fallbackText
        ? fallbackText
        : typeof rawText === "string"
        ? rawText
        : "",
  };
}

function fallbackParse(text, fallbackText = "") {
  if (!text || typeof text !== "string") {
    return buildFallbackObject(text, fallbackText);
  }

  const hallucinations_found = /HALLUCINATIONS FOUND:\s*YES/i.test(text);

  const typesMatch = text.match(/TYPES:\s*(.*)/i);
  const justificationMatch = text.match(
    /JUSTIFICATION:\s*([\s\S]*?)CORRECTED ANSWER:/i
  );
  const correctedMatch = text.match(/CORRECTED ANSWER:\s*([\s\S]*)/i);

  const parsedTypes = typesMatch
    ? typesMatch[1]
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
        .filter((s) => s !== "[]")
    : [];

  const corrected_answer = correctedMatch
    ? correctedMatch[1].trim()
    : fallbackText || text;

  return {
    hallucinations_found,
    types: parsedTypes,
    justification: justificationMatch ? justificationMatch[1].trim() : "",
    corrected_answer,
  };
}

function normalizeReviewerOutput(rawText, fallbackText = "") {
  if (!rawText || typeof rawText !== "string") {
    return buildFallbackObject(rawText, fallbackText);
  }

  const cleaned = rawText
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  try {
    const parsed = JSON.parse(cleaned);

    const correctedAnswer =
      typeof parsed.corrected_answer === "string" &&
      parsed.corrected_answer.trim()
        ? parsed.corrected_answer
        : fallbackText || "";

    return {
      hallucinations_found: Boolean(parsed.hallucinations_found),
      types: Array.isArray(parsed.types)
        ? parsed.types.filter((item) => typeof item === "string")
        : [],
      justification:
        typeof parsed.justification === "string" ? parsed.justification : "",
      corrected_answer: correctedAnswer,
    };
  } catch (err) {
    console.warn("⚠️ JSON parsing failed. Using fallback parser.");
    return fallbackParse(rawText, fallbackText);
  }
}

// ================== MAIN ==================

async function main() {
  const config = await loadConfig(CONFIG_PATH);
  validateConfig(config);

  const runTimeUtc = new Date().toISOString();
  const runId = sanitizeRunId(runTimeUtc);

  const cases =
    config.task_type === "generation"
      ? [{ id: "generation_task", input: "N/A" }]
      : config.cases;

  const caseResults = [];
  for (const caseConfig of cases) {
    const caseResult = await runCase(caseConfig, config);
    caseResults.push(caseResult);
  }

  let modelSequence = [];

  if (config.pipeline.architecture === "sequential") {
    modelSequence = buildModelSequence(config.pipeline.models);
  } else if (config.pipeline.architecture === "consensus") {
    const gens = config.pipeline.consensus.generators;
    const agg = config.pipeline.consensus.aggregator;

    modelSequence = [
      ...gens.map((m, i) => `${m.model} (generator_${i + 1})`),
      `${agg.model} (aggregator)`,
    ];
  }

  const runResult = {
    run_id: runId,
    run_time_utc: runTimeUtc,
    architecture: config?.pipeline?.architecture || "sequential",
    model_sequence: modelSequence,
    cases: caseResults,
  };

  await writeResults(runResult, runId);

  // 🔹 NEW: generate markdown
  const md = buildMarkdown(runResult);

  await fs.mkdir(RESULTS_MD_DIR, { recursive: true });
  await fs.mkdir(HISTORY_MD_DIR, { recursive: true });

  await fs.writeFile(path.join(RESULTS_MD_DIR, "latest.md"), md);
  await fs.writeFile(path.join(HISTORY_MD_DIR, `${runId}.md`), md);

  console.log(`Run complete: ${runId}`);
}

// ================== CONFIG ==================

async function loadConfig(configPath) {
  const raw = await fs.readFile(configPath, "utf8");
  return YAML.parse(raw);
}

function validateConfig(config) {
  if (!config.pipeline) {
    throw new Error("Missing pipeline config");
  }

  if (config.task_type === "evaluation") {
    if (!config.cases || config.cases.length === 0) {
      throw new Error("Evaluation tasks require cases");
    }
  }
}

// ================== (rest of file unchanged) ==================
