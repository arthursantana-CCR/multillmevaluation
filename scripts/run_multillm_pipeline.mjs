import fs from "fs/promises";
import path from "path";
import process from "process";
import YAML from "yaml";

const CONFIG_PATH = path.resolve("eval_config.yaml");
const RESULTS_DIR = path.resolve("results");
const HISTORY_DIR = path.resolve("results/history");
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
    return resolveKnowledgePlaceholders(config.task, config); // 🔹 NEW
  }

  if (config.task_type === "evaluation") {
    const task = resolveKnowledgePlaceholders(config.task, config); // 🔹 NEW
    return `${task}\n\n${JSON.stringify(caseConfig)}`;
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
      typeof parsed.corrected_answer === "string" && parsed.corrected_answer.trim()
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

function normalizeAggregatorOutput(rawText, fallbackText = "") {
  if (!rawText || typeof rawText !== "string") {
    return {
      sources_used: [],
      hallucinations_found: false,
      types: [],
      justification: "Fallback: unable to parse aggregator output.",
      corrected_answer: fallbackText || "",
    };
  }

  const cleaned = rawText
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  try {
    const parsed = JSON.parse(cleaned);

    return {
      sources_used: Array.isArray(parsed.sources_used)
        ? parsed.sources_used.filter((item) => typeof item === "string")
        : [],
      hallucinations_found: Boolean(parsed.hallucinations_found),
      types: Array.isArray(parsed.types)
        ? parsed.types.filter((item) => typeof item === "string")
        : [],
      justification:
        typeof parsed.justification === "string" ? parsed.justification : "",
      corrected_answer:
        typeof parsed.corrected_answer === "string" && parsed.corrected_answer.trim()
          ? parsed.corrected_answer
          : fallbackText || "",
    };

  } catch (err) {
    console.warn("⚠️ Aggregator JSON parsing failed. Using fallback.");
    return {
      sources_used: [],
      hallucinations_found: false,
      types: [],
      justification: "Fallback: unable to parse aggregator output.",
      corrected_answer: fallbackText || rawText,
    };
  }
}

function resolveKnowledgePlaceholders(text, config) {
  if (!text || typeof text !== "string") return text;

  // Existing: {{ccr.key}}
  text = text.replace(/\{\{ccr\.(\w+)\}\}/g, (match, key) => {
    const knowledge = config.knowledge?.ccr?.[key];
    if (!knowledge) {
      console.warn(`⚠️ Unknown knowledge placeholder: {{ccr.${key}}}`);
      return match;
    }
    return YAML.stringify(knowledge);
  });

  // 🔹 NEW: {{lesson_plan}}
  text = text.replace(/\{\{lesson_plan\}\}/g, () => {
    const knowledge = config.knowledge?.lesson_plan;
    if (!knowledge) {
      console.warn(`⚠️ Unknown knowledge placeholder: {{lesson_plan}}`);
      return "{{lesson_plan}}";
    }
    return YAML.stringify(knowledge);
  });

  return text;
}

async function loadKnowledge(config) {
  if (typeof config.hallucination_rubric === "string" && config.hallucination_rubric.endsWith(".yaml")) {
    const raw = await fs.readFile(path.resolve(config.hallucination_rubric), "utf8");
    const parsed = YAML.parse(raw);
    config.hallucination_rubric = parsed.rubric;
  }

  // 🔹 NEW: load lesson_plan knowledge file
  if (typeof config.knowledge?.lesson_plan === "string" && config.knowledge.lesson_plan.endsWith(".yaml")) {
    const raw = await fs.readFile(path.resolve(config.knowledge.lesson_plan), "utf8");
    config.knowledge.lesson_plan = YAML.parse(raw);
  }

  if (config.knowledge?.ccr) {
    for (const [key, filePath] of Object.entries(config.knowledge.ccr)) {
      if (typeof filePath === "string" && filePath.endsWith(".yaml")) {
        const raw = await fs.readFile(path.resolve(filePath), "utf8");
        config.knowledge.ccr[key] = YAML.parse(raw);
      }
    }
  }
}

// ================== MAIN ==================

async function main() {
  const config = await loadConfig(CONFIG_PATH);
  validateConfig(config);
  await loadKnowledge(config);

  const runTimeUtc = new Date().toISOString();
  const runId = sanitizeRunId(runTimeUtc);

  // 🔹 NEW: allow generation tasks to run without user-defined cases
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
    const hc = config.pipeline.consensus.hallucination_checker;

    modelSequence = [
      ...gens.map((m, i) => `${m.model} (generator_${i + 1})`),
      `${agg.model} (aggregator)`,
      `${hc.model} (hallucination_checker)`,
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

  await import("./render_results.mjs");

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

  if (config.pipeline.architecture === "consensus") {
    if (!config.pipeline.consensus?.hallucination_checker) {
      throw new Error("Consensus architecture requires a hallucination_checker entry under pipeline.consensus");
    }
  }
}

// ================== CASE ROUTER ==================

async function runCase(caseConfig, config) {
  const architecture = config?.pipeline?.architecture || "sequential";

  if (architecture === "consensus") {
    return runConsensusCase(caseConfig, config);
  }

  return runSequentialCase(caseConfig, config);
}

// ================== SEQUENTIAL ==================

async function runSequentialCase(caseConfig, config) {
  const seq = config.pipeline.models;

  const generator = seq.find((m) => m.role === "generator");
  const r1 = seq.find((m) => m.role === "reviewer_1");
  const r2 = seq.find((m) => m.role === "reviewer_2");
  const rf = seq.find((m) => m.role === "final_reviewer");

  const generatorPrompt = buildGeneratorPrompt(caseConfig, config);

  const generatorRaw = await callModel({
    provider: generator.provider,
    model: generator.model,
    systemInstruction: generatorPrompt.systemInstruction,
    userPrompt: generatorPrompt.userPrompt,
    parameters: config.parameters,
  });

  let currentValidOutput = generatorRaw;

  const reviewer1Result = await callReviewerWithRetry({
    provider: r1.provider,
    model: r1.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({
      config,
      previousOutput: currentValidOutput,
    }),
    parameters: config.parameters,
    fallbackText: currentValidOutput,
    config,
  });

  if (reviewer1Result.status === "success") {
    const normalized = reviewer1Result.parsed_review;
    if (!normalized.corrected_answer) {
      console.warn("⚠️ Missing corrected_answer. Using previous valid output.");
    } else {
      currentValidOutput = normalized.corrected_answer;
    }
  }

  const reviewer2Result = await callReviewerWithRetry({
    provider: r2.provider,
    model: r2.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({
      config,
      previousOutput: currentValidOutput,
    }),
    parameters: config.parameters,
    fallbackText: currentValidOutput,
    config,
  });

  if (reviewer2Result.status === "success") {
    const normalized = reviewer2Result.parsed_review;
    if (!normalized.corrected_answer) {
      console.warn("⚠️ Missing corrected_answer. Using previous valid output.");
    } else {
      currentValidOutput = normalized.corrected_answer;
    }
  }

  const finalResult = await callReviewerWithRetry({
    provider: rf.provider,
    model: rf.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({
      config,
      previousOutput: currentValidOutput,
    }),
    parameters: config.parameters,
    fallbackText: currentValidOutput,
    config,
  });

  if (finalResult.status === "success") {
    const normalized = finalResult.parsed_review;
    if (!normalized.corrected_answer) {
      console.warn("⚠️ Missing corrected_answer. Using previous valid output.");
    } else {
      currentValidOutput = normalized.corrected_answer;
    }
  }

  return {
    case_id: caseConfig.id,
    prompt: generatorPrompt.userPrompt,
    model_sequence: buildModelSequence(seq),
    outputs: {
      generator_output: {
        status: "success",
        raw_text: generatorRaw,
      },
      reviewer_1_output: reviewer1Result,
      reviewer_2_output: reviewer2Result,
      final_reviewer_output: finalResult,
      final_output: currentValidOutput,
    },
  };
}

// ================== CONSENSUS ==================

async function runConsensusCase(caseConfig, config) {
  const { generators, aggregator, hallucination_checker } = config.pipeline.consensus;

  const taskInput = buildTaskInput({ config, caseConfig });

  const candidateOutputs = await Promise.all(
    generators.map((m) =>
      callModel({
        provider: m.provider,
        model: m.model,
        systemInstruction: "",
        userPrompt: `
You are a helpful AI assistant.

Your task is to generate a response based on the instructions provided.

Do NOT perform evaluation, analysis, or structured reporting.

Focus only on producing the best possible final answer.

TASK:
${taskInput}
`,
        parameters: config.parameters,
      })
    )
  );

  const [c1, c2, c3] = candidateOutputs;

  // ── STEP 1: Aggregator — synthesis only ──
  const aggregationPrompt = `
You are the aggregator in a multi-model evaluation pipeline.

Your task is to evaluate THREE candidate answers and synthesize ONE final response that combines the best parts of all candidates.

Focus purely on quality, completeness, and coherence.
Do NOT perform hallucination checking — that will be handled separately.
Do NOT output JSON. Output only the final synthesized response as plain text.

--------------------------------------------------
TASK
--------------------------------------------------
${taskInput}

--------------------------------------------------
CANDIDATES
--------------------------------------------------
A:
${c1}

B:
${c2}

C:
${c3}

--------------------------------------------------
REQUIRED PROCESS
--------------------------------------------------

PHASE 1 — EVALUATE EACH CANDIDATE

For each candidate (A, B, C):

1. Usability check
A candidate is NOT usable if:
- it is an error message
- it is empty
- it does not attempt to answer the task
- it is mostly malformed or nonsensical

2. Quality check
For each usable candidate, assess:
- Adherence to the task prompt
- Completeness
- Clarity and formatting
- Pedagogical usefulness

PHASE 2 — SYNTHESIZE THE FINAL RESPONSE

- Draw the best sections from each candidate
- If one candidate is clearly strongest across all dimensions, you may use it as-is
- Do NOT combine sections in a way that creates contradictions or inconsistencies
- Do NOT add new content not present in any candidate
- The final response must fully satisfy the original task

PHASE 3 — OUTPUT

Structure your output in two parts:

PART 1 — SYNTHESIS ANALYSIS
Provide a brief structured analysis of your synthesis decisions:
- For each candidate (A, B, C), state roughly what percentage it contributed to the final response and which specific sections or strengths it contributed
- Explain why certain sections were preferred over others
- Note any sections that were excluded and why
Keep this analysis to 150-200 words maximum.

PART 2 — SYNTHESIZED RESPONSE
Output the complete final synthesized response as plain text.

Separate the two parts with this exact delimiter on its own line:
---SYNTHESIS BEGIN---

Do NOT output JSON.
`;

const aggregatorRaw = await callModel({
    provider: aggregator.provider,
    model: aggregator.model,
    systemInstruction: "",
    userPrompt: aggregationPrompt,
    parameters: config.parameters,
  });

  // ── Guard: aggregator empty or error ──
  if (!aggregatorRaw || aggregatorRaw.trim() === "" || isModelError(aggregatorRaw)) {
    console.warn("⚠️ Aggregator returned empty or error. Falling back to candidate 1.");
    return {
      case_id: caseConfig.id,
      prompt: taskInput,
      model_sequence: [],
      outputs: {
        candidate_1: { raw_text: c1 },
        candidate_2: { raw_text: c2 },
        candidate_3: { raw_text: c3 },
        aggregator_output: { raw_text: aggregatorRaw, error: "empty_or_failed" },
        hallucination_check_output: null,
        final_output: c1,
      },
    };
  }

  // ── Split analysis from synthesized response ──
  const delimiter = "---SYNTHESIS BEGIN---";
  const delimiterIndex = aggregatorRaw.indexOf(delimiter);
  const synthesizedText = delimiterIndex !== -1
    ? aggregatorRaw.slice(delimiterIndex + delimiter.length).trim()
    : aggregatorRaw;

  // ── Guard: synthesized text empty after split ──
  if (!synthesizedText || synthesizedText.trim() === "") {
    console.warn("⚠️ Synthesized text is empty after delimiter split. Falling back to candidate 1.");
    return {
      case_id: caseConfig.id,
      prompt: taskInput,
      model_sequence: [],
      outputs: {
        candidate_1: { raw_text: c1 },
        candidate_2: { raw_text: c2 },
        candidate_3: { raw_text: c3 },
        aggregator_output: { raw_text: aggregatorRaw, error: "delimiter_missing_or_empty_synthesis" },
        hallucination_check_output: null,
        final_output: c1,
      },
    };
  }

  // ── STEP 2: Hallucination checker ──
  const hallucinationResult = await callReviewerWithRetry({
    provider: hallucination_checker.provider,
    model: hallucination_checker.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({
      config,
      previousOutput: synthesizedText,
    }),
    parameters: config.parameters,
    fallbackText: synthesizedText,
    config,
  });

  const finalOutput =
    hallucinationResult.status === "success"
      ? hallucinationResult.parsed_review.corrected_answer || synthesizedText
      : synthesizedText;

  return {
    case_id: caseConfig.id,
    prompt: taskInput,
    model_sequence: [],
    outputs: {
      candidate_1: { raw_text: c1 },
      candidate_2: { raw_text: c2 },
      candidate_3: { raw_text: c3 },
      aggregator_output: { raw_text: aggregatorRaw },
      hallucination_check_output: hallucinationResult,
      final_output: finalOutput,
    },
  };
}

// ================== RETRY ==================

async function callReviewerWithRetry(args) {
  const raw = await callModel({ ...args, userPrompt: args.baseUserPrompt });

  if (isModelError(raw)) {
    return {
      status: "failed",
      raw_text: raw,
      error_type: detectErrorType(raw),
      parsed_review: {
        hallucinations_found: false,
        types: [],
        justification: "",
        corrected_answer: args.fallbackText || "",
      },
    };
  }

  const normalized = normalizeReviewerOutput(raw, args.fallbackText);

  return {
    status: "success",
    raw_text: raw,
    parsed_review: normalized,
  };
}

function parseReviewerOutput(rawText, fallbackText = "") {
  return normalizeReviewerOutput(rawText, fallbackText);
}

// ================== PROMPTS ==================
function buildReviewerPrompt({ config, previousOutput }) {
  let prompt = `
You are a reviewer in a multi-step AI evaluation pipeline.

Your task is to evaluate the following answer for hallucinations and correct it if necessary.

IMPORTANT:
- You MUST follow the required JSON output format exactly
- Do NOT include any text outside the JSON
- If no hallucinations are found, return the original answer unchanged

ANSWER TO REVIEW:
${previousOutput}
`;

  if (config.hallucination_rubric) {
    prompt += `\n\nHALLUCINATION RUBRIC:\n${config.hallucination_rubric}`;
  }

  if (config.output_format?.template) {
    prompt += `\n\nOUTPUT FORMAT:\n${config.output_format.template}`;
  }

  return prompt;
}

function buildGeneratorPrompt(caseConfig, config) {
  const generatorSystemInstruction = `
You are an expert educator.

Your task is to generate a complete, high-quality response based on the instructions provided.

IMPORTANT:
- Do NOT perform evaluation or hallucination analysis
- Do NOT output JSON
- Output ONLY the final lesson plan as free text
- Do NOT include any structured metadata

Focus on clarity, completeness, and pedagogical quality.
`;

if (config.task_type === "generation") {
  return {
    systemInstruction: generatorSystemInstruction,
    userPrompt: resolveKnowledgePlaceholders(config.task, config), // 🔹 NEW
  };
}

return {
  systemInstruction: generatorSystemInstruction,
  userPrompt: resolveKnowledgePlaceholders(`${config.task}\n\n${JSON.stringify(caseConfig)}`, config), // 🔹 NEW
};
}

// ================== MODEL CALLS ==================

async function callModel(args) {
  if (args.provider === "openai") return callOpenAI(args);
  if (args.provider === "anthropic") return callAnthropic(args);
  if (args.provider === "google") return callGemini(args);
  throw new Error(`Unknown provider: ${args.provider}`);
}

async function callOpenAI({ model, systemInstruction, userPrompt, parameters }) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemInstruction },
        { role: "user", content: userPrompt },
      ],
      temperature: parameters.temperature,
      max_tokens: parameters.max_tokens,
    }),
  });

  const data = await res.json();
  const text = data.choices?.[0]?.message?.content || "";

  if (typeof text === "string" && text && !text.trim().startsWith("{")) {
    console.warn("⚠️ Non-JSON output detected from OpenAI model");
  }

  return text;
}

async function callAnthropic({ model, systemInstruction, userPrompt, parameters }) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      system: systemInstruction,
      messages: [{ role: "user", content: userPrompt }],
      max_tokens: parameters.max_tokens,
    }),
  });

  const data = await res.json();
  const text = data.content?.[0]?.text || "";

  if (typeof text === "string" && text && !text.trim().startsWith("{")) {
    console.warn("⚠️ Non-JSON output detected from Anthropic model");
  }

  return text;
}

async function callGemini({ model, systemInstruction, userPrompt, parameters }) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`;

  const body = {
    contents: [
      {
        role: "user",
        parts: [{ text: userPrompt }],
      },
    ],
    generationConfig: {
      temperature: parameters?.temperature ?? 0,
      maxOutputTokens: parameters?.max_tokens ?? 1024,
    },
  };

  if (systemInstruction) {
    body.systemInstruction = {
      parts: [{ text: systemInstruction }],
    };
  }

  const maxAttempts = 4;
  let lastError = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      const data = await res.json();

      if (!res.ok) {
        if (isRetryableStatus(res.status) && attempt < maxAttempts) {
          const delay = Math.min(1000 * 2 ** (attempt - 1), 8000);
          const jitter = Math.floor(Math.random() * 300);
          console.warn(
            `Gemini retryable error ${res.status} on attempt ${attempt}/${maxAttempts}. Retrying in ${delay + jitter}ms`
          );
          await sleep(delay + jitter);
          continue;
        }

        const errorMessage =
          data?.error?.message ||
          `Gemini API error ${res.status}: ${JSON.stringify(data)}`;

        console.error("Gemini API error:", JSON.stringify(data, null, 2));
        return `[ERROR: ${errorMessage}]`;
      }

      if (data.promptFeedback?.blockReason) {
        console.error("Gemini prompt blocked:", JSON.stringify(data, null, 2));
        return "[ERROR: Gemini prompt blocked]";
      }

      const candidate = data.candidates?.[0];

      if (!candidate) {
        console.error("Gemini returned no candidates:", JSON.stringify(data, null, 2));
        return "[ERROR: Gemini returned no candidates]";
      }

      if (candidate.finishReason && candidate.finishReason !== "STOP") {
        console.error(
          "Gemini candidate did not finish normally:",
          JSON.stringify(data, null, 2)
        );
        return `[ERROR: Gemini finish reason: ${candidate.finishReason}]`;
      }

      const text = (candidate.content?.parts || [])
        .map((part) => part.text || "")
        .join("")
        .trim();

      if (!text) {
        console.error("Gemini returned empty text:", JSON.stringify(data, null, 2));
        return "[ERROR: Gemini empty text]";
      }

      if (!text.trim().startsWith("{")) {
        console.warn("⚠️ Non-JSON output detected from Gemini model");
      }

      return text;
    } catch (err) {
      lastError = err;

      if (attempt < maxAttempts) {
        const delay = Math.min(1000 * 2 ** (attempt - 1), 8000);
        const jitter = Math.floor(Math.random() * 300);
        console.warn(
          `Gemini request failed on attempt ${attempt}/${maxAttempts}: ${err.message}. Retrying in ${delay + jitter}ms`
        );
        await sleep(delay + jitter);
        continue;
      }
    }
  }

  return `[ERROR: ${lastError?.message || "Gemini request failed"}]`;
}

// ================== IO ==================

async function writeResults(runResult, runId) {
  await fs.mkdir(RESULTS_DIR, { recursive: true });
  await fs.mkdir(HISTORY_DIR, { recursive: true });

  const pretty = JSON.stringify(runResult, null, 2);

  await fs.writeFile(path.join(RESULTS_DIR, "latest.json"), pretty);
  await fs.writeFile(path.join(HISTORY_DIR, `${runId}.json`), pretty);
}

function sanitizeRunId(iso) {
  return iso.replace(/[:.]/g, "-");
}

main().catch((err) => {
  console.error("FATAL ERROR:", err);
  process.exit(1);
});
