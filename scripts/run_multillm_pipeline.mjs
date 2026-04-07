import fs from "fs/promises";
import path from "path";
import process from "process";
import YAML from "yaml";

const CONFIG_PATH = path.resolve("eval_config.yaml");
const RESULTS_DIR = path.resolve("results");
const HISTORY_DIR = path.resolve("results/history");

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
  console.log(`Cases processed: ${caseResults.length}`);
}

async function loadConfig(configPath) {
  const raw = await fs.readFile(configPath, "utf8");
  return YAML.parse(raw);
}

// ---------------- VALIDATION ----------------

function validateReviewerOutput(parsed) {
  const issues = [];

  if (parsed.hallucinations_found === null) {
    issues.push("missing_hallucination_flag");
  }

  if (!Array.isArray(parsed.types)) {
    issues.push("types_not_array");
  }

  if (!parsed.justification || parsed.justification.length < 5) {
    issues.push("empty_or_short_justification");
  }

  if (!parsed.corrected_answer || parsed.corrected_answer.length < 10) {
    issues.push("missing_or_short_corrected_answer");
  }

  // Truncation heuristic
  if (!/[.!?]$/.test(parsed.corrected_answer.trim())) {
    issues.push("possible_truncation");
  }

  return {
    is_valid: issues.length === 0,
    issues,
  };
}

// ---------------- MAIN CASE ----------------

async function runCase(caseConfig, config) {
  const sequence = config.pipeline.models;
  const generatorModel = sequence.find((m) => m.role === "generator");
  const reviewer1Model = sequence.find((m) => m.role === "reviewer_1");
  const reviewer2Model = sequence.find((m) => m.role === "reviewer_2");
  const finalReviewerModel = sequence.find((m) => m.role === "final_reviewer");

  const generatorPrompt = buildGeneratorPrompt(caseConfig, config);
  const promptText = generatorPrompt.userPrompt;

  const generatorRaw = await callModel({
    provider: generatorModel.provider,
    model: generatorModel.model,
    systemInstruction: generatorPrompt.systemInstruction,
    userPrompt: generatorPrompt.userPrompt,
    parameters: config.parameters,
  });

  // ---------------- REVIEWER 1 ----------------

  const reviewer1Raw = await callModel({
    provider: reviewer1Model.provider,
    model: reviewer1Model.model,
    systemInstruction: config.system_instruction,
    userPrompt: buildReviewerPrompt({
      caseConfig,
      config,
      previousRole: generatorModel.role,
      previousModel: generatorModel.model,
      previousOutput: generatorRaw,
    }).userPrompt,
    parameters: config.parameters,
  });

  const reviewer1Parsed = parseReviewerOutput(reviewer1Raw, generatorRaw);
  const reviewer1Validation = validateReviewerOutput(reviewer1Parsed);

  // ---------------- REVIEWER 2 ----------------

  const reviewer2Raw = await callModel({
    provider: reviewer2Model.provider,
    model: reviewer2Model.model,
    systemInstruction: config.system_instruction,
    userPrompt: buildReviewerPrompt({
      caseConfig,
      config,
      previousRole: reviewer1Model.role,
      previousModel: reviewer1Model.model,
      previousOutput: reviewer1Parsed.corrected_answer,
    }).userPrompt,
    parameters: config.parameters,
  });

  const reviewer2Parsed = parseReviewerOutput(
    reviewer2Raw,
    reviewer1Parsed.corrected_answer
  );
  const reviewer2Validation = validateReviewerOutput(reviewer2Parsed);

  // ---------------- FINAL REVIEWER ----------------

  const finalReviewerRaw = await callModel({
    provider: finalReviewerModel.provider,
    model: finalReviewerModel.model,
    systemInstruction: config.system_instruction,
    userPrompt: buildReviewerPrompt({
      caseConfig,
      config,
      previousRole: reviewer2Model.role,
      previousModel: reviewer2Model.model,
      previousOutput: reviewer2Parsed.corrected_answer,
    }).userPrompt,
    parameters: config.parameters,
  });

  const finalReviewerParsed = parseReviewerOutput(
    finalReviewerRaw,
    reviewer2Parsed.corrected_answer
  );
  const finalValidation = validateReviewerOutput(finalReviewerParsed);

  const finalOutput =
    finalReviewerParsed.corrected_answer ||
    reviewer2Parsed.corrected_answer ||
    reviewer1Parsed.corrected_answer ||
    generatorRaw;

  return {
    case_id: caseConfig.id,
    input: {
      prompt_text: promptText,
      ...extractCaseInput(caseConfig),
    },
    outputs: {
      generator_output: {
        role: generatorModel.role,
        provider: generatorModel.provider,
        model: generatorModel.model,
        raw_text: generatorRaw,
      },
      reviewer_1_output: {
        role: reviewer1Model.role,
        provider: reviewer1Model.provider,
        model: reviewer1Model.model,
        raw_text: reviewer1Raw,
        parsed_review: {
          ...reviewer1Parsed,
          validation: reviewer1Validation,
        },
      },
      reviewer_2_output: {
        role: reviewer2Model.role,
        provider: reviewer2Model.provider,
        model: reviewer2Model.model,
        raw_text: reviewer2Raw,
        parsed_review: {
          ...reviewer2Parsed,
          validation: reviewer2Validation,
        },
      },
      final_reviewer_output: {
        role: finalReviewerModel.role,
        provider: finalReviewerModel.provider,
        model: finalReviewerModel.model,
        raw_text: finalReviewerRaw,
        parsed_review: {
          ...finalReviewerParsed,
          validation: finalValidation,
        },
      },
      final_output: finalOutput,
    },
  };
}

// ---------------- PARSER (FIXED) ----------------

function parseReviewerOutput(rawText, fallbackText = "") {
  const text = String(rawText ?? "").trim();
  const fallback = String(fallbackText ?? "").trim();

  const hallucinations = extractHeader(text, "HALLUCINATIONS FOUND");
  const typesRaw = extractHeader(text, "TYPES");
  const justification = extractSection(text, "JUSTIFICATION", "CORRECTED ANSWER");
  const corrected = extractSection(text, "CORRECTED ANSWER");

  return {
    hallucinations_found: normalizeHallucinations(hallucinations),
    types: normalizeTypes(typesRaw),
    justification: justification || "",
    corrected_answer: corrected || fallback,
  };
}

function extractHeader(text, header) {
  const match = text.match(new RegExp(`${header}:\\s*(.+)`, "i"));
  return match?.[1]?.trim() || "";
}

function extractSection(text, start, end = null) {
  const startRegex = new RegExp(`${start}:`, "i");
  const startMatch = startRegex.exec(text);
  if (!startMatch) return "";

  const rest = text.slice(startMatch.index + startMatch[0].length);

  if (!end) return rest.trim();

  const endRegex = new RegExp(`${end}:`, "i");
  const endMatch = endRegex.exec(rest);

  return endMatch ? rest.slice(0, endMatch.index).trim() : rest.trim();
}

function normalizeHallucinations(value) {
  if (/yes/i.test(value)) return true;
  if (/no/i.test(value)) return false;
  if (/^\d+$/.test(value)) return Number(value) > 0;
  return null;
}

function normalizeTypes(value) {
  if (!value || value === "[]" || /none/i.test(value)) return [];
  return value.split(",").map((v) => v.trim()).filter(Boolean);
}

// ---------------- UTIL ----------------

function extractCaseInput(caseConfig) {
  const input = {};
  for (const [key, value] of Object.entries(caseConfig)) {
    if (key === "id") continue;
    input[key] = value;
  }
  return input;
}

function buildGeneratorPrompt(caseConfig, config) {
  const caseFields = formatCaseFields(caseConfig);
  return {
    systemInstruction: config.system_instruction,
    userPrompt: `You are the generator...\n\n${config.task}\n\n${caseFields}`,
  };
}

function buildReviewerPrompt({ caseConfig, config, previousRole, previousModel, previousOutput }) {
  const caseFields = formatCaseFields(caseConfig);
  return {
    systemInstruction: config.system_instruction,
    userPrompt: `Review the answer.\n\n${config.hallucination_rubric}\n\nPrevious Output:\n${previousOutput}`,
  };
}

function formatCaseFields(caseConfig) {
  return Object.entries(caseConfig)
    .filter(([k]) => k !== "id")
    .map(([k, v]) => `${k}:\n${v}`)
    .join("\n\n");
}

// ---------------- IO ----------------

async function writeResults(runResult, runId, runTimeUtc) {
  await fs.mkdir(RESULTS_DIR, { recursive: true });
  await fs.mkdir(HISTORY_DIR, { recursive: true });

  const pretty = JSON.stringify(runResult, null, 2);

  await fs.writeFile(path.join(RESULTS_DIR, "latest.json"), pretty);
  await fs.writeFile(path.join(RESULTS_DIR, "latest_timestamp.txt"), runTimeUtc);
  await fs.writeFile(path.join(HISTORY_DIR, `${runId}.json`), pretty);
}

function buildRunResult({ config, runId, runTimeUtc, caseResults }) {
  return {
    run_id: runId,
    run_time_utc: runTimeUtc,
    description: config.description,
    cases: caseResults,
  };
}

function sanitizeRunId(isoString) {
  return isoString.replace(/[:.]/g, "-");
}

main().catch((error) => {
  console.error("Pipeline failed.");
  console.error(error);
  process.exit(1);
});
