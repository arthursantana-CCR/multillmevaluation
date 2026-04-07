import fs from "fs/promises";
import path from "path";
import process from "process";
import YAML from "yaml";

const CONFIG_PATH = path.resolve("eval_config.yaml");
const RESULTS_DIR = path.resolve("results");
const HISTORY_DIR = path.resolve("results/history");
const MAX_RETRIES = 1;

// ================== SCORING ==================

function scoreHallucinations(types) {
  const weights = {
    fabrication: 2,
    unsupported_inference: 1,
    contradiction: 2,
    irrelevant: 0.5,
  };

  let score = 0;

  for (const type of types || []) {
    const key = String(type).toLowerCase().replace(/\s+/g, "_");
    score += weights[key] ?? 1;
  }

  return score;
}

function classifySeverity(score) {
  if (score === 0) return "none";
  if (score <= 1) return "low";
  if (score <= 3) return "medium";
  return "high";
}

// ================== MAIN ==================

async function main() {
  const config = await loadConfig(CONFIG_PATH);
  validateConfig(config);

  const runTimeUtc = new Date().toISOString();
  const runId = sanitizeRunId(runTimeUtc);

  const caseResults = [];
  for (const caseConfig of config.cases) {
    const caseResult = await runCase(caseConfig, config);
    caseResults.push(caseResult);
  }

  const runResult = buildRunResult({
    config,
    runId,
    runTimeUtc,
    caseResults,
  });

  await writeResults(runResult, runId, runTimeUtc);

  console.log(`Run complete: ${runId}`);
}

// ================== CONFIG ==================

async function loadConfig(configPath) {
  const raw = await fs.readFile(configPath, "utf8");
  return YAML.parse(raw);
}

function validateConfig(config) {
  if (!config.pipeline || !config.pipeline.models) {
    throw new Error("Missing pipeline config");
  }
}

// ================== CASE ==================

async function runCase(caseConfig, config) {
  const sequence = config.pipeline.models;

  const generator = sequence.find((m) => m.role === "generator");
  const r1 = sequence.find((m) => m.role === "reviewer_1");
  const r2 = sequence.find((m) => m.role === "reviewer_2");
  const rf = sequence.find((m) => m.role === "final_reviewer");

  const generatorPrompt = buildGeneratorPrompt(caseConfig, config);

  const generatorRaw = await callModel({
    provider: generator.provider,
    model: generator.model,
    systemInstruction: generatorPrompt.systemInstruction,
    userPrompt: generatorPrompt.userPrompt,
    parameters: config.parameters,
  });

  // ---------- REVIEWER 1 ----------

  const reviewer1Result = await callReviewerWithRetry({
    provider: r1.provider,
    model: r1.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({
      config,
      previousOutput: generatorRaw,
    }),
    parameters: config.parameters,
    fallbackText: generatorRaw,
  });

  const r1Score = scoreHallucinations(reviewer1Result.parsed_review.types);
  reviewer1Result.parsed_review.score = r1Score;
  reviewer1Result.parsed_review.severity = classifySeverity(r1Score);

  // ---------- REVIEWER 2 ----------

  const reviewer2Result = await callReviewerWithRetry({
    provider: r2.provider,
    model: r2.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({
      config,
      previousOutput: reviewer1Result.parsed_review.corrected_answer,
    }),
    parameters: config.parameters,
    fallbackText: reviewer1Result.parsed_review.corrected_answer,
  });

  const r2Score = scoreHallucinations(reviewer2Result.parsed_review.types);
  reviewer2Result.parsed_review.score = r2Score;
  reviewer2Result.parsed_review.severity = classifySeverity(r2Score);

  // ---------- FINAL REVIEWER ----------

  const finalResult = await callReviewerWithRetry({
    provider: rf.provider,
    model: rf.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({
      config,
      previousOutput: reviewer2Result.parsed_review.corrected_answer,
    }),
    parameters: config.parameters,
    fallbackText: reviewer2Result.parsed_review.corrected_answer,
  });

  const finalScore = scoreHallucinations(finalResult.parsed_review.types);
  finalResult.parsed_review.score = finalScore;
  finalResult.parsed_review.severity = classifySeverity(finalScore);

  const finalOutput =
    finalResult.parsed_review.corrected_answer ||
    reviewer2Result.parsed_review.corrected_answer ||
    reviewer1Result.parsed_review.corrected_answer ||
    generatorRaw;

  const pipelineMetrics = {
    reviewer_1_score: r1Score,
    reviewer_2_score: r2Score,
    final_score: finalScore,
    improvement: r1Score - finalScore,
  };

  return {
    case_id: caseConfig.id,
    outputs: {
      generator_output: generatorRaw,
      reviewer_1_output: reviewer1Result,
      reviewer_2_output: reviewer2Result,
      final_reviewer_output: finalResult,
      final_output: finalOutput,
      pipeline_metrics: pipelineMetrics,
    },
  };
}

// ================== RETRY ==================

async function callReviewerWithRetry({
  provider,
  model,
  systemInstruction,
  baseUserPrompt,
  parameters,
  fallbackText,
}) {
  let prompt = baseUserPrompt;

  for (let i = 0; i <= MAX_RETRIES; i++) {
    const raw = await callModel({
      provider,
      model,
      systemInstruction,
      userPrompt: prompt,
      parameters,
    });

    const parsed = parseReviewerOutput(raw, fallbackText);
    const validation = validateReviewerOutput(parsed);

    if (validation.is_valid) {
      return { raw_text: raw, parsed_review: parsed };
    }

    prompt = buildRetryPrompt(prompt, validation.issues, raw);
  }

  return {
    raw_text: "",
    parsed_review: {
      hallucinations_found: false,
      types: [],
      justification: "Fallback used",
      corrected_answer: fallbackText,
    },
  };
}

function buildRetryPrompt(original, issues, raw) {
  return `${original}

RETRY REQUIRED.
Issues:
${issues.join("\n")}

Fix formatting and output VALID JSON + summary.
Previous output:
${raw}`;
}

// ================== PARSER ==================

function parseReviewerOutput(raw, fallback) {
  const json = extractJSON(raw);

  if (json) return normalizeJSON(json, fallback);

  return {
    hallucinations_found: false,
    types: [],
    justification: "",
    corrected_answer: fallback,
  };
}

function extractJSON(text) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;

  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

function normalizeJSON(j, fallback) {
  return {
    hallucinations_found: Boolean(j.hallucinations_found),
    types: Array.isArray(j.types) ? j.types : [],
    justification: j.justification || "",
    corrected_answer: j.corrected_answer || fallback,
  };
}

// ================== VALIDATION ==================

function validateReviewerOutput(p) {
  const issues = [];

  if (typeof p.hallucinations_found !== "boolean") issues.push("bad_flag");
  if (!Array.isArray(p.types)) issues.push("bad_types");
  if (!p.corrected_answer) issues.push("missing_answer");

  return { is_valid: issues.length === 0, issues };
}

// ================== PROMPTS ==================

function buildGeneratorPrompt(caseConfig, config) {
  return {
    systemInstruction: config.system_instruction,
    userPrompt: `${config.task}\n\n${JSON.stringify(caseConfig)}`,
  };
}

function buildReviewerPrompt({ config, previousOutput }) {
  return `
Review the answer.

OUTPUT FORMAT:

1) JSON:
{
 "hallucinations_found": boolean,
 "types": string[],
 "justification": string,
 "corrected_answer": string
}

2) SUMMARY

Previous Output:
${previousOutput}

Rubric:
${config.hallucination_rubric}
`;
}

// ================== MODEL ==================

async function callModel({ provider, model, systemInstruction, userPrompt, parameters }) {
  // simplified for clarity (same as your previous implementation)
  return "Mock response"; // replace with real calls
}

// ================== IO ==================

async function writeResults(runResult, runId, runTimeUtc) {
  await fs.mkdir(RESULTS_DIR, { recursive: true });
  await fs.mkdir(HISTORY_DIR, { recursive: true });

  const pretty = JSON.stringify(runResult, null, 2);

  await fs.writeFile(path.join(RESULTS_DIR, "latest.json"), pretty);
  await fs.writeFile(path.join(HISTORY_DIR, `${runId}.json`), pretty);
}

function buildRunResult({ config, runId, runTimeUtc, caseResults }) {
  return {
    run_id: runId,
    run_time_utc: runTimeUtc,
    cases: caseResults,
  };
}

function sanitizeRunId(iso) {
  return iso.replace(/[:.]/g, "-");
}

main().catch(console.error);
