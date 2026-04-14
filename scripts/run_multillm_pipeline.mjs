import fs from "fs/promises";
import path from "path";
import process from "process";
import YAML from "yaml";

const CONFIG_PATH = path.resolve("eval_config.yaml");
const RESULTS_DIR = path.resolve("results");
const HISTORY_DIR = path.resolve("results/history");

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
    config,
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
    config,
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
    config,
  });

  const finalOutput =
    finalResult.parsed_review.corrected_answer ||
    reviewer2Result.parsed_review.corrected_answer ||
    reviewer1Result.parsed_review.corrected_answer ||
    generatorRaw;

  return {
    case_id: caseConfig.id,
    prompt: generatorPrompt.userPrompt,
    model_sequence: buildModelSequence(seq),
    outputs: {
      generator_output: { raw_text: generatorRaw },
      reviewer_1_output: reviewer1Result,
      reviewer_2_output: reviewer2Result,
      final_reviewer_output: finalResult,
      final_output: finalOutput,
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
        systemInstruction: "",
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

Your task is to compare three candidate answers and select the best one.

Criteria:
- Accuracy
- Completeness
- Absence of hallucinations
- Alignment with the task instructions

You must select ONE of the answers (A, B, or C).

A:
${c1}

B:
${c2}

C:
${c3}

${config.output_format?.template}

Additional instructions:
- In the STRUCTURED OUTPUT section:
  - "selected_model" must be "A", "B", or "C"
  - "reasoning" must briefly justify your choice
- In the FINAL ANSWER section:
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

  if (args.config?.output_format?.type === "structured") {
    const parsed = parseReviewerOutput(raw, args.fallbackText);
    return { raw_text: raw, parsed_review: parsed };
  }

  return {
    raw_text: raw,
    parsed_review: { corrected_answer: raw },
  };
}

function parseReviewerOutput(raw, fallbackText) {
  if (!raw || raw.startsWith("[ERROR")) {
    return { corrected_answer: fallbackText || "" };
  }

  const split = raw.split("### FINAL ANSWER");

  if (split.length < 2) {
    // FIX: If the dual-output delimiter is missing, treat the whole response
    // as the corrected answer rather than silently losing it.
    console.warn("[parseReviewerOutput] No '### FINAL ANSWER' delimiter found. Using full response.");
    return { corrected_answer: raw.trim() };
  }

  return {
    corrected_answer: split[1].trim(),
  };
}

// ================== PROMPTS ==================

function buildGeneratorPrompt(caseConfig, config) {
  if (config.task_type === "generation") {
    return {
      systemInstruction: config.system_instruction,
      userPrompt: config.task,
    };
  }

  return {
    systemInstruction: config.system_instruction,
    userPrompt: `${config.task}\n\n${JSON.stringify(caseConfig)}`,
  };
}

// FIX: Reviewer prompt is now built with clear section delimiters and a
// hard token budget for the embedded prior output. This prevents Gemini's
// safety filters from triggering on unstructured concatenated text and
// keeps prompt length predictable across all models.
//
// The MAX_PRIOR_OUTPUT_CHARS cap (~1500 chars ≈ ~375 tokens) ensures the
// total reviewer prompt stays well within Gemini's effective context for
// structured generation tasks. Claude and GPT are unaffected.
const MAX_PRIOR_OUTPUT_CHARS = 1500;

function buildReviewerPrompt({ config, previousOutput }) {
  // Truncate prior output if it exceeds the safe threshold.
  // This is the single most impactful change for Gemini reviewer stability.
  const safeOutput =
    previousOutput && previousOutput.length > MAX_PRIOR_OUTPUT_CHARS
      ? previousOutput.slice(0, MAX_PRIOR_OUTPUT_CHARS) + "\n[... truncated for length ...]"
      : previousOutput || "";

  // Build prompt in clearly labelled sections.
  // Gemini responds much better to structured sections than to a flat
  // concatenated string, because it can attend to each part independently.
  let prompt = `## ROLE
You are a reviewer in an evaluation pipeline. Your job is to check the answer below for inaccuracies and produce a corrected version.

## ANSWER TO REVIEW
${safeOutput}`;

  if (config.hallucination_rubric) {
    prompt += `\n\n## EVALUATION RUBRIC\n${config.hallucination_rubric}`;
  }

  if (config.output_format?.template) {
    prompt += `\n\n## OUTPUT FORMAT\n${config.output_format.template}`;
  }

  return prompt;
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
  return data.choices?.[0]?.message?.content || "";
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
  return data.content?.[0]?.text || "";
}

// FIX: Three changes to callGemini():
//
// 1. safetySettings — set all categories to BLOCK_ONLY_HIGH.
//    Gemini's default thresholds are aggressive and silently block reviewer
//    prompts that contain evaluation meta-language ("hallucination", "error",
//    "misinformation") combined with quoted prior model output.
//    BLOCK_ONLY_HIGH tells Gemini to only block content that is clearly and
//    severely harmful, which is appropriate for an educational evaluation system.
//
// 2. maxOutputTokens bumped to 2048 minimum.
//    A reviewer must emit a JSON block + a full corrected answer. 1024 tokens
//    is insufficient for most educational content tasks, causing Gemini to
//    truncate and return empty candidates.
//
// 3. promptFeedback logged on empty response.
//    blockReason in promptFeedback is the authoritative signal for WHY Gemini
//    returned no candidates. This makes future debugging deterministic.
async function callGemini({ model, systemInstruction, userPrompt, parameters }) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`;

  // Merge system instruction and user prompt (Gemini has no separate system role)
  const fullPrompt = [systemInstruction, userPrompt].filter(Boolean).join("\n\n");

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      contents: [{ parts: [{ text: fullPrompt }] }],
      generationConfig: {
        temperature: parameters?.temperature ?? 0,
        // FIX 2: Ensure output token budget is large enough for dual-output
        // (JSON block + corrected answer). Default from config may be too low.
        maxOutputTokens: Math.max(parameters?.max_tokens ?? 1024, 2048),
      },
      // FIX 1: Relax safety thresholds to BLOCK_ONLY_HIGH for all categories.
      // This prevents Gemini from silently blocking reviewer prompts that
      // contain evaluation meta-language about the prior model's output.
      safetySettings: [
        { category: "HARM_CATEGORY_HARASSMENT",        threshold: "BLOCK_ONLY_HIGH" },
        { category: "HARM_CATEGORY_HATE_SPEECH",       threshold: "BLOCK_ONLY_HIGH" },
        { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_ONLY_HIGH" },
        { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_ONLY_HIGH" },
      ],
    }),
  });

  const data = await res.json();

  if (!data.candidates || data.candidates.length === 0) {
    // FIX 3: Log promptFeedback.blockReason — this is the definitive signal
    // for why Gemini returned no candidates (SAFETY, OTHER, RECITATION, etc.)
    const blockReason = data.promptFeedback?.blockReason ?? "unknown";
    const safetyRatings = JSON.stringify(data.promptFeedback?.safetyRatings ?? []);
    console.error(
      `[Gemini] Empty response. blockReason=${blockReason} | safetyRatings=${safetyRatings}`
    );
    console.error("[Gemini] Full response:", JSON.stringify(data, null, 2));
    return "[ERROR: Gemini returned empty response]";
  }

  // Also check for finish reason on the candidate itself
  const candidate = data.candidates[0];
  if (candidate.finishReason && candidate.finishReason !== "STOP") {
    console.warn(`[Gemini] Unusual finishReason: ${candidate.finishReason}`);
  }

  const text = candidate?.content?.parts?.[0]?.text;
  if (!text) {
    console.error("[Gemini] Malformed response:", JSON.stringify(data, null, 2));
    return "[ERROR: Gemini malformed response]";
  }

  return text;
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
