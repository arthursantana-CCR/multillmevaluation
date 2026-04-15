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

  // 🔥 NEW: strip markdown code fences
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
  const { generators, aggregator } = config.pipeline.consensus;

  const generatorPrompt = `
${config.task}

${JSON.stringify(caseConfig)}
`;

  const candidateOutputs = await Promise.all(
    generators.map((m) =>
      callModel({
        provider: m.provider,
        model: m.model,
        systemInstruction: "", // important for Gemini
        userPrompt: `
You are a helpful AI assistant.

Your task is to generate a complete and high-quality response based on the instructions provided.

Do NOT perform evaluation, analysis, or structured reporting.

Focus only on producing the best possible final answer.

TASK:
${config.task}

INPUT:
${JSON.stringify(caseConfig)}
`,
        parameters: config.parameters,
      })
    )
  );

  const [c1, c2, c3] = candidateOutputs;

  const aggregationPrompt = `
You are an evaluator in a multi-model system.

Your task is to evaluate THREE candidate answers and select the best one.

IMPORTANT:
You must evaluate EACH candidate before making a decision.

---

STEP 1 — Evaluate each candidate

For EACH answer (A, B, C), consider:

1. Is the answer usable?
   - If it contains an error message (e.g., "[ERROR: ...]"), it is NOT usable

2. Does it contain hallucinations?
   - Apply the hallucination rubric strictly

3. Overall quality:
   - completeness
   - clarity
   - pedagogical usefulness
   - alignment with the task

---

STEP 2 — Selection rules

Follow this priority order:

1. Prefer answers that are usable (not error messages)
2. Prefer answers with fewer hallucinations
3. If multiple answers are similar in hallucination level:
   → choose the one with higher overall quality

IMPORTANT:
- Do NOT select an answer that is an error message unless ALL answers are errors
- Do NOT assume an answer is correct just because it is detailed

---

STEP 3 — Output

You must return ONLY valid JSON in this format:

{
  "selected_model": "A" | "B" | "C",
  "justification": "<clear reasoning explaining your selection>",
  "hallucinations_found": <true or false>,
  "types": <array>,
  "corrected_answer": "<final selected answer>"
}

---

CANDIDATES:

A:
${c1}

B:
${c2}

C:
${c3}

${config.output_format?.template}

Additional instructions:
- In the output JSON:
  - "selected_model" must be "A", "B", or "C"
  - "reasoning" must briefly justify your choice
- In "corrected_answer":
  - Output ONLY the selected answer
  - Do NOT modify it unless necessary to fix critical issues
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
    model_sequence: [],
    outputs: {
      candidate_1: { raw_text: c1 },
      candidate_2: { raw_text: c2 },
      candidate_3: { raw_text: c3 },
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
      userPrompt: config.task,
    };
  }

  return {
    systemInstruction: generatorSystemInstruction,
    userPrompt: `${config.task}\n\n${JSON.stringify(caseConfig)}`,
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

export async function runPipelineWithConfig(config) {
  validateConfig(config);

  const runTimeUtc = new Date().toISOString();
  const runId = runTimeUtc.replace(/[:.]/g, "-");

  const caseResults = [];
  for (const caseConfig of config.cases) {
    const caseResult = await runCase(caseConfig, config);
    caseResults.push(caseResult);
  }

  return {
    run_id: runId,
    run_time_utc: runTimeUtc,
    architecture: config?.pipeline?.architecture || "sequential",
    cases: caseResults
  };
}
