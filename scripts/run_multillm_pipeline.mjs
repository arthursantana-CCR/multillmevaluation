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
    currentValidOutput =
      reviewer1Result.parsed_review.corrected_answer || currentValidOutput;
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
    currentValidOutput =
      reviewer2Result.parsed_review.corrected_answer || currentValidOutput;
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
    currentValidOutput =
      finalResult.parsed_review.corrected_answer || currentValidOutput;
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

  if (isModelError(raw)) {
    return {
      status: "failed",
      raw_text: raw,
      error_type: detectErrorType(raw),
      parsed_review: {
        corrected_answer: args.fallbackText || "",
      },
    };
  }

  if (args.config?.output_format?.type === "structured") {
    const parsed = parseReviewerOutput(raw, args.fallbackText);

    return {
      status: "success",
      raw_text: raw,
      parsed_review: parsed,
    };
  }

  return {
    status: "success",
    raw_text: raw,
    parsed_review: { corrected_answer: raw },
  };
}

function parseReviewerOutput(raw, fallbackText) {
  if (!raw) {
    return { corrected_answer: fallbackText || "" };
  }

  const split = raw.split("### FINAL ANSWER");

  if (split.length < 2) {
    return { corrected_answer: raw };
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

function buildReviewerPrompt({ config, previousOutput }) {
  let prompt = `Review the following answer for hallucinations, unsupported claims, or omissions. Then produce the required output.

ANSWER TO REVIEW:
${previousOutput}`;

  if (config.hallucination_rubric) {
    prompt += `\n\nRubric:\n${config.hallucination_rubric}`;
  }

  if (config.output_format?.template) {
    prompt += `\n\n${config.output_format.template}`;
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
