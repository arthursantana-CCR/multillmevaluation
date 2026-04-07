import fs from "fs/promises";
import path from "path";
import process from "process";
import YAML from "yaml";

const CONFIG_PATH = path.resolve("eval_config.yaml");
const RESULTS_DIR = path.resolve("results");
const HISTORY_DIR = path.resolve("results/history");
const MAX_RETRIES = 1;

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

function validateConfig(config) {
  if (!config || typeof config !== "object") {
    throw new Error("Invalid config: expected a YAML object.");
  }

  if (!config.description || typeof config.description !== "string") {
    throw new Error("Invalid config: 'description' is required.");
  }

  if (!config.pipeline || !Array.isArray(config.pipeline.models) || config.pipeline.models.length === 0) {
    throw new Error("Invalid config: 'pipeline.models' must be a non-empty array.");
  }

  const allowedRoles = new Set(["generator", "reviewer_1", "reviewer_2", "final_reviewer"]);
  const seenRoles = new Set();

  for (const modelDef of config.pipeline.models) {
    if (!modelDef.role || !allowedRoles.has(modelDef.role)) {
      throw new Error(
        `Invalid config: each pipeline model must have a valid 'role'. Got: ${modelDef.role ?? "undefined"}`
      );
    }
    if (seenRoles.has(modelDef.role)) {
      throw new Error(`Invalid config: duplicate pipeline role '${modelDef.role}'.`);
    }
    seenRoles.add(modelDef.role);

    if (!modelDef.provider || typeof modelDef.provider !== "string") {
      throw new Error(`Invalid config: model '${modelDef.role}' is missing 'provider'.`);
    }
    if (!modelDef.model || typeof modelDef.model !== "string") {
      throw new Error(`Invalid config: model '${modelDef.role}' is missing 'model'.`);
    }
  }

  for (const requiredRole of allowedRoles) {
    if (!seenRoles.has(requiredRole)) {
      throw new Error(`Invalid config: missing required pipeline role '${requiredRole}'.`);
    }
  }

  if (!config.parameters || typeof config.parameters !== "object") {
    throw new Error("Invalid config: 'parameters' is required.");
  }

  if (!config.system_instruction || typeof config.system_instruction !== "string") {
    throw new Error("Invalid config: 'system_instruction' is required.");
  }

  if (!config.task_rubric || typeof config.task_rubric !== "string") {
    throw new Error("Invalid config: 'task_rubric' is required.");
  }

  if (!config.hallucination_rubric || typeof config.hallucination_rubric !== "string") {
    throw new Error("Invalid config: 'hallucination_rubric' is required.");
  }

  if (!config.task || typeof config.task !== "string") {
    throw new Error("Invalid config: 'task' is required.");
  }

  if (!Array.isArray(config.cases) || config.cases.length === 0) {
    throw new Error("Invalid config: 'cases' must be a non-empty array.");
  }

  for (const caseConfig of config.cases) {
    if (!caseConfig.id || typeof caseConfig.id !== "string") {
      throw new Error("Invalid config: every case must have an 'id'.");
    }
  }
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

  if (!endsLikeCompleteText(parsed.corrected_answer)) {
    issues.push("possible_truncation");
  }

  return {
    is_valid: issues.length === 0,
    issues,
  };
}

function endsLikeCompleteText(text) {
  const value = String(text ?? "").trim();
  if (!value) return false;
  return /[.!?}\]"']$/.test(value);
}

// ---------------- RETRY ----------------

async function callReviewerWithRetry({
  provider,
  model,
  systemInstruction,
  baseUserPrompt,
  parameters,
  fallbackText,
}) {
  const attempts = [];
  let currentPrompt = baseUserPrompt;

  for (let attemptNumber = 0; attemptNumber <= MAX_RETRIES; attemptNumber += 1) {
    const rawText = await callModel({
      provider,
      model,
      systemInstruction,
      userPrompt: currentPrompt,
      parameters,
    });

    const parsed = parseReviewerOutput(rawText, fallbackText);
    const validation = validateReviewerOutput(parsed);

    attempts.push({
      attempt_number: attemptNumber + 1,
      raw_text: rawText,
      parsed_review: parsed,
      validation,
      was_retry: attemptNumber > 0,
    });

    if (validation.is_valid) {
      return {
        raw_text: rawText,
        parsed_review: parsed,
        validation,
        attempts,
        used_fallback: false,
      };
    }

    if (attemptNumber < MAX_RETRIES) {
      currentPrompt = buildRetryPrompt({
        originalPrompt: baseUserPrompt,
        invalidRawText: rawText,
        validationIssues: validation.issues,
      });
    }
  }

  const fallbackParsed = {
    hallucinations_found: false,
    types: [],
    justification:
      "Fallback applied because reviewer output remained invalid after retry.",
    corrected_answer: String(fallbackText ?? "").trim(),
  };
  const fallbackValidation = {
    is_valid: false,
    issues: ["fallback_used_after_failed_retries"],
  };

  attempts.push({
    attempt_number: attempts.length + 1,
    raw_text: "",
    parsed_review: fallbackParsed,
    validation: fallbackValidation,
    was_retry: false,
    is_fallback: true,
  });

  return {
    raw_text: attempts[attempts.length - 2]?.raw_text ?? "",
    parsed_review: fallbackParsed,
    validation: fallbackValidation,
    attempts,
    used_fallback: true,
  };
}

function buildRetryPrompt({ originalPrompt, invalidRawText, validationIssues }) {
  return [
    originalPrompt.trim(),
    "",
    "-----------------------------------",
    "RETRY INSTRUCTION",
    "",
    "Your previous output was INVALID.",
    "",
    "Detected issues:",
    ...validationIssues.map((issue) => `- ${issue}`),
    "",
    "You MUST regenerate your answer using the EXACT required format.",
    "Do not omit any section.",
    "Do not rename headers.",
    "CORRECTED ANSWER must always be present and complete.",
    "If no hallucinations are found, return TYPES: [] and include the unchanged answer under CORRECTED ANSWER.",
    "Do not output any commentary outside the required format.",
    "",
    "Previous invalid output:",
    invalidRawText.trim(),
  ].join("\n");
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

  const reviewer1Prompt = buildReviewerPrompt({
    caseConfig,
    config,
    previousRole: generatorModel.role,
    previousModel: generatorModel.model,
    previousOutput: generatorRaw,
  });

  const reviewer1Result = await callReviewerWithRetry({
    provider: reviewer1Model.provider,
    model: reviewer1Model.model,
    systemInstruction: reviewer1Prompt.systemInstruction,
    baseUserPrompt: reviewer1Prompt.userPrompt,
    parameters: config.parameters,
    fallbackText: generatorRaw,
  });

  // ---------------- REVIEWER 2 ----------------

  const reviewer2Prompt = buildReviewerPrompt({
    caseConfig,
    config,
    previousRole: reviewer1Model.role,
    previousModel: reviewer1Model.model,
    previousOutput: reviewer1Result.parsed_review.corrected_answer,
  });

  const reviewer2Result = await callReviewerWithRetry({
    provider: reviewer2Model.provider,
    model: reviewer2Model.model,
    systemInstruction: reviewer2Prompt.systemInstruction,
    baseUserPrompt: reviewer2Prompt.userPrompt,
    parameters: config.parameters,
    fallbackText: reviewer1Result.parsed_review.corrected_answer,
  });

  // ---------------- FINAL REVIEWER ----------------

  const finalReviewerPrompt = buildReviewerPrompt({
    caseConfig,
    config,
    previousRole: reviewer2Model.role,
    previousModel: reviewer2Model.model,
    previousOutput: reviewer2Result.parsed_review.corrected_answer,
  });

  const finalReviewerResult = await callReviewerWithRetry({
    provider: finalReviewerModel.provider,
    model: finalReviewerModel.model,
    systemInstruction: finalReviewerPrompt.systemInstruction,
    baseUserPrompt: finalReviewerPrompt.userPrompt,
    parameters: config.parameters,
    fallbackText: reviewer2Result.parsed_review.corrected_answer,
  });

  const finalOutput =
    finalReviewerResult.parsed_review.corrected_answer ||
    reviewer2Result.parsed_review.corrected_answer ||
    reviewer1Result.parsed_review.corrected_answer ||
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
        raw_text: reviewer1Result.raw_text,
        parsed_review: {
          ...reviewer1Result.parsed_review,
          validation: reviewer1Result.validation,
        },
        retry_summary: {
          attempts: reviewer1Result.attempts.length,
          used_fallback: reviewer1Result.used_fallback,
        },
        attempt_history: reviewer1Result.attempts,
      },
      reviewer_2_output: {
        role: reviewer2Model.role,
        provider: reviewer2Model.provider,
        model: reviewer2Model.model,
        raw_text: reviewer2Result.raw_text,
        parsed_review: {
          ...reviewer2Result.parsed_review,
          validation: reviewer2Result.validation,
        },
        retry_summary: {
          attempts: reviewer2Result.attempts.length,
          used_fallback: reviewer2Result.used_fallback,
        },
        attempt_history: reviewer2Result.attempts,
      },
      final_reviewer_output: {
        role: finalReviewerModel.role,
        provider: finalReviewerModel.provider,
        model: finalReviewerModel.model,
        raw_text: finalReviewerResult.raw_text,
        parsed_review: {
          ...finalReviewerResult.parsed_review,
          validation: finalReviewerResult.validation,
        },
        retry_summary: {
          attempts: finalReviewerResult.attempts.length,
          used_fallback: finalReviewerResult.used_fallback,
        },
        attempt_history: finalReviewerResult.attempts,
      },
      final_output: finalOutput,
    },
  };
}

// ---------------- PARSER ----------------

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
  const escaped = escapeRegex(header);
  const match = text.match(new RegExp(`^\\s*${escaped}\\s*:\\s*(.+)\\s*$`, "im"));
  return match?.[1]?.trim() || "";
}

function extractSection(text, start, end = null) {
  const startRegex = new RegExp(`^\\s*${escapeRegex(start)}\\s*:\\s*`, "im");
  const startMatch = startRegex.exec(text);
  if (!startMatch) return "";

  const rest = text.slice(startMatch.index + startMatch[0].length);

  if (!end) return rest.trim();

  const endRegex = new RegExp(`^\\s*${escapeRegex(end)}\\s*:`, "im");
  const endMatch = endRegex.exec(rest);

  return endMatch ? rest.slice(0, endMatch.index).trim() : rest.trim();
}

function normalizeHallucinations(value) {
  const raw = String(value ?? "").trim();
  if (/^yes$/i.test(raw)) return true;
  if (/^no$/i.test(raw)) return false;
  if (/^\d+$/.test(raw)) return Number(raw) > 0;
  return null;
}

function normalizeTypes(value) {
  const raw = String(value ?? "").trim();
  if (!raw || raw === "[]" || /^none$/i.test(raw)) return [];
  return raw
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean)
    .filter((v) => v !== "[]");
}

function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ---------------- PROMPTS / UTIL ----------------

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
    userPrompt: [
      "You are the generator in a multi-LLM evaluation pipeline.",
      "",
      "Use the rubric and task below to produce the initial answer.",
      "Return only the requested output format.",
      "",
      "Task Rubric:",
      config.task_rubric.trim(),
      "",
      "Task:",
      config.task.trim(),
      "",
      "Case Input:",
      caseFields,
    ].join("\n"),
  };
}

function buildReviewerPrompt({ caseConfig, config, previousRole, previousModel, previousOutput }) {
  const caseFields = formatCaseFields(caseConfig);

  return {
    systemInstruction: config.system_instruction,
    userPrompt: [
      "You are a reviewer/editor in a multi-LLM evaluation pipeline.",
      "",
      "Your job is to review the previous answer for hallucinations according to the rubric below.",
      "If hallucinations are found, fully correct them.",
      "If no hallucinations are found, keep the answer substantively the same and return it as the corrected answer.",
      "",
      "Hallucination Rubric:",
      config.hallucination_rubric.trim(),
      "",
      "Original Task Rubric:",
      config.task_rubric.trim(),
      "",
      "Original Task:",
      config.task.trim(),
      "",
      "Case Input:",
      caseFields,
      "",
      `Previous Stage Role: ${previousRole}`,
      `Previous Stage Model: ${previousModel}`,
      "",
      "Previous Output:",
      previousOutput,
    ].join("\n"),
  };
}

function formatCaseFields(caseConfig) {
  return Object.entries(caseConfig)
    .filter(([k]) => k !== "id")
    .map(([k, v]) => {
      if (typeof v === "string") {
        return `${k}:\n${v}`;
      }
      return `${k}:\n${JSON.stringify(v, null, 2)}`;
    })
    .join("\n\n");
}

// ---------------- MODEL CALLS ----------------

async function callModel({ provider, model, systemInstruction, userPrompt, parameters }) {
  switch (provider) {
    case "openai":
      return callOpenAI({ model, systemInstruction, userPrompt, parameters });
    case "anthropic":
      return callAnthropic({ model, systemInstruction, userPrompt, parameters });
    case "google":
      return callGemini({ model, systemInstruction, userPrompt, parameters });
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

async function callOpenAI({ model, systemInstruction, userPrompt, parameters }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("Missing OPENAI_API_KEY.");
  }

  const body = {
    model,
    messages: [
      {
        role: "system",
        content: systemInstruction,
      },
      {
        role: "user",
        content: userPrompt,
      },
    ],
    temperature: parameters.temperature,
    max_tokens: parameters.max_tokens,
  };

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status} ${JSON.stringify(data)}`);
  }

  const text = data?.choices?.[0]?.message?.content;
  if (!text) {
    throw new Error("OpenAI API returned no message content.");
  }

  return text.trim();
}

async function callAnthropic({ model, systemInstruction, userPrompt, parameters }) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("Missing ANTHROPIC_API_KEY.");
  }

  const body = {
    model,
    system: systemInstruction,
    messages: [
      {
        role: "user",
        content: userPrompt,
      },
    ],
    temperature: parameters.temperature,
    max_tokens: parameters.max_tokens,
  };

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(`Anthropic API error: ${response.status} ${JSON.stringify(data)}`);
  }

  const text = Array.isArray(data?.content)
    ? data.content
        .filter((item) => item?.type === "text")
        .map((item) => item.text)
        .join("\n")
    : "";

  if (!text) {
    throw new Error("Anthropic API returned no text content.");
  }

  return text.trim();
}

async function callGemini({ model, systemInstruction, userPrompt, parameters }) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Missing GEMINI_API_KEY.");
  }

  const normalizedModel = normalizeGeminiModelName(model);
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${normalizedModel}:generateContent`;

  const body = {
    systemInstruction: {
      parts: [{ text: systemInstruction }],
    },
    contents: [
      {
        role: "user",
        parts: [{ text: userPrompt }],
      },
    ],
    generationConfig: {
      temperature: parameters.temperature,
      topP: parameters.top_p,
      maxOutputTokens: parameters.max_tokens,
    },
  };

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(`Gemini API error: ${response.status} ${JSON.stringify(data)}`);
  }

  const text = data?.candidates?.[0]?.content?.parts
    ?.map((part) => part?.text || "")
    .join("\n")
    .trim();

  if (!text) {
    throw new Error("Gemini API returned no text content.");
  }

  return text;
}

function normalizeGeminiModelName(model) {
  return model.startsWith("models/") ? model.slice("models/".length) : model;
}

// ---------------- IO ----------------

async function writeResults(runResult, runId, runTimeUtc) {
  await fs.mkdir(RESULTS_DIR, { recursive: true });
  await fs.mkdir(HISTORY_DIR, { recursive: true });

  const pretty = JSON.stringify(runResult, null, 2);

  await fs.writeFile(path.join(RESULTS_DIR, "latest.json"), pretty, "utf8");
  await fs.writeFile(path.join(RESULTS_DIR, "latest_timestamp.txt"), runTimeUtc, "utf8");
  await fs.writeFile(path.join(HISTORY_DIR, `${runId}.json`), pretty, "utf8");
}

function buildRunResult({ config, runId, runTimeUtc, caseResults }) {
  return {
    run_id: runId,
    run_time_utc: runTimeUtc,
    description: config.description,
    pipeline: {
      model_sequence: config.pipeline.models.map((item) => ({
        role: item.role,
        provider: item.provider,
        model: item.model,
      })),
      parameters: {
        temperature: config.parameters.temperature,
        top_p: config.parameters.top_p,
        max_tokens: config.parameters.max_tokens,
      },
      retry_policy: {
        max_retries_per_reviewer: MAX_RETRIES,
      },
    },
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
