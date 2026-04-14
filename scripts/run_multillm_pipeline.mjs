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
        systemInstruction: config.system_instruction,
        userPrompt: generatorPrompt,
        parameters: config.parameters,
      })
    )
  );

  const [c1, c2, c3] = candidateOutputs;

  const aggregationPrompt = `
Compare the answers and select the best one.

Return JSON ONLY:

{
  "selected_model": "A | B | C",
  "reasoning": "...",
  "final_answer": {...}
}

A:
${c1}

B:
${c2}

C:
${c3}
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
  return {
    corrected_answer: raw || fallbackText || "",
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
  if (config.task_type === "generation") {
    return previousOutput;
  }

  return `${previousOutput}\n\nRubric:\n${config.hallucination_rubric}`;
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

async function callGemini({ model, userPrompt }) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`;

  const res = await fetch(url, {
    method: "POST",
    body: JSON.stringify({
      contents: [{ parts: [{ text: userPrompt }] }],
    }),
  });

  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || "";
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
