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

// ================== HELPERS ==================

function buildModelSequence(sequence) {
  return sequence.map((m) => `${m.model} (${m.role || "candidate"})`);
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

  const runResult = {
    run_id: runId,
    run_time_utc: runTimeUtc,
    architecture: config?.pipeline?.architecture || "sequential",
    model_sequence: buildModelSequence(config.pipeline.models),
    cases: caseResults,
  };

  await writeResults(runResult, runId);
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

  const promptUsed = generatorPrompt.userPrompt;
  const modelSequence = buildModelSequence(seq);

  const generatorRaw = await callModel({
    provider: generator.provider,
    model: generator.model,
    systemInstruction: generatorPrompt.systemInstruction,
    userPrompt: generatorPrompt.userPrompt,
    parameters: config.parameters,
  });

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

  const finalOutput =
    finalResult.parsed_review.corrected_answer ||
    reviewer2Result.parsed_review.corrected_answer ||
    reviewer1Result.parsed_review.corrected_answer ||
    generatorRaw;

  return {
    case_id: caseConfig.id,
    prompt: promptUsed,
    model_sequence: modelSequence,
    outputs: {
      generator_output: {
        raw_text: generatorRaw,
        model: generator.model,
        provider: generator.provider,
      },
      reviewer_1_output: reviewer1Result,
      reviewer_2_output: reviewer2Result,
      final_reviewer_output: finalResult,
      final_output: finalOutput,
    },
  };
}

// ================== CONSENSUS ==================

async function runConsensusCase(caseConfig, config) {
  const consensusConfig = config.pipeline.consensus;

  if (!consensusConfig || !consensusConfig.generators || !consensusConfig.aggregator) {
    throw new Error("Missing consensus configuration in eval_config.yaml");
  }

  const generators = consensusConfig.generators;
  const aggregator = consensusConfig.aggregator;

  const generatorPrompt = `
You are an expert evaluator.

Your task is to complete the task below.

IMPORTANT:
- Do NOT analyze hallucinations
- Do NOT critique answers
- Do NOT act as a reviewer
- Only perform the task

TASK:
${config.task}

INPUT:
${JSON.stringify(caseConfig)}

Return ONLY the final answer in the required format.
`;

  const modelSequence = [
    ...generators.map((g) => `${g.model} (candidate)`),
    `${aggregator.model} (aggregator)`,
  ];

  const candidateOutputs = await Promise.all(
    generators.map((m) =>
      callModel({
        provider: m.provider,
        model: m.model,
        systemInstruction: config.system_instruction,
        userPrompt: generatorPrompt,
        parameters: config.parameters,
      })
    )
  );

  const [c1, c2, c3] = candidateOutputs;

  // ✅ UPDATED AGGREGATOR PROMPT (SMART + STRUCTURED)
  const aggregationPrompt = `
You are an expert evaluator.

Your task is to compare multiple answers and select the BEST one.

You MUST follow these evaluation criteria:
1) Factual accuracy
2) Logical consistency
3) Completeness
4) Absence of hallucinations

Hallucination rubric:
${JSON.stringify(config.hallucination_rubric || {}, null, 2)}

---

Answer A:
${c1}

Answer B:
${c2}

Answer C:
${c3}

---

Instructions:

1) Analyze each answer carefully
2) Identify hallucinations (if any)
3) Compare quality across answers
4) Select the BEST answer

---

Output format (STRICT JSON ONLY):

{
  "selected_model": "A | B | C",
  "reasoning": "Short explanation of why this answer is best",
  "final_answer": "<FULL selected answer>"
}

DO NOT include anything outside JSON.
`;

  const finalOutput = await callModel({
    provider: aggregator.provider,
    model: aggregator.model,
    systemInstruction: config.system_instruction,
    userPrompt: aggregationPrompt,
    parameters: config.parameters,
  });

  return {
    case_id: caseConfig.id,
    prompt: generatorPrompt,
    model_sequence: modelSequence,
    outputs: {
      generator_output: {
        raw_text: "",
        model: "",
        provider: "",
      },

      reviewer_1_output: { raw_text: c1 },
      reviewer_2_output: { raw_text: c2 },
      final_reviewer_output: { raw_text: c3 },

      final_output: finalOutput,
    },
  };
}

// ================== RETRY ==================

async function callReviewerWithRetry(args) {
  let prompt = args.baseUserPrompt;

  for (let i = 0; i <= MAX_RETRIES; i++) {
    const raw = await callModel({
      ...args,
      userPrompt: prompt,
    });

    const parsed = parseReviewerOutput(raw, args.fallbackText);
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
      corrected_answer: args.fallbackText,
    },
  };
}

function buildRetryPrompt(original, issues, raw) {
  return `${original}

RETRY REQUIRED:
${issues.join("\n")}

Fix output format strictly.

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
Return JSON + summary.

JSON schema:
{
 "hallucinations_found": boolean,
 "types": string[],
 "justification": string,
 "corrected_answer": string
}

Previous Output:
${previousOutput}

Rubric:
${config.hallucination_rubric}
`;
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
  return data.choices[0].message.content;
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
  return data.content[0].text;
}

async function callGemini({ model, userPrompt }) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`;

  const res = await fetch(url, {
    method: "POST",
    body: JSON.stringify({
      contents: [{ parts: [{ text: userPrompt }] }],
    }),
  });

  const data = await res.json();
  return data.candidates[0].content.parts[0].text;
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

main().catch(console.error);
