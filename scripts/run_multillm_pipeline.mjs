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

  const reviewer1Prompt = buildReviewerPrompt({
    caseConfig,
    config,
    previousRole: generatorModel.role,
    previousModel: generatorModel.model,
    previousOutput: generatorRaw,
  });

  const reviewer1Raw = await callModel({
    provider: reviewer1Model.provider,
    model: reviewer1Model.model,
    systemInstruction: reviewer1Prompt.systemInstruction,
    userPrompt: reviewer1Prompt.userPrompt,
    parameters: config.parameters,
  });

  const reviewer1Parsed = parseReviewerOutput(reviewer1Raw);

  const reviewer2Prompt = buildReviewerPrompt({
    caseConfig,
    config,
    previousRole: reviewer1Model.role,
    previousModel: reviewer1Model.model,
    previousOutput: reviewer1Parsed.corrected_answer || reviewer1Raw,
  });

  const reviewer2Raw = await callModel({
    provider: reviewer2Model.provider,
    model: reviewer2Model.model,
    systemInstruction: reviewer2Prompt.systemInstruction,
    userPrompt: reviewer2Prompt.userPrompt,
    parameters: config.parameters,
  });

  const reviewer2Parsed = parseReviewerOutput(reviewer2Raw);

  const finalReviewerPrompt = buildReviewerPrompt({
    caseConfig,
    config,
    previousRole: reviewer2Model.role,
    previousModel: reviewer2Model.model,
    previousOutput: reviewer2Parsed.corrected_answer || reviewer2Raw,
  });

  const finalReviewerRaw = await callModel({
    provider: finalReviewerModel.provider,
    model: finalReviewerModel.model,
    systemInstruction: finalReviewerPrompt.systemInstruction,
    userPrompt: finalReviewerPrompt.userPrompt,
    parameters: config.parameters,
  });

  const finalReviewerParsed = parseReviewerOutput(finalReviewerRaw);
  const finalOutput = finalReviewerParsed.corrected_answer || finalReviewerRaw;

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
        parsed_review: reviewer1Parsed,
      },
      reviewer_2_output: {
        role: reviewer2Model.role,
        provider: reviewer2Model.provider,
        model: reviewer2Model.model,
        raw_text: reviewer2Raw,
        parsed_review: reviewer2Parsed,
      },
      final_reviewer_output: {
        role: finalReviewerModel.role,
        provider: finalReviewerModel.provider,
        model: finalReviewerModel.model,
        raw_text: finalReviewerRaw,
        parsed_review: finalReviewerParsed,
      },
      final_output: finalOutput,
    },
  };
}

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
  const lines = [];
  for (const [key, value] of Object.entries(caseConfig)) {
    if (key === "id") continue;

    if (typeof value === "string") {
      lines.push(`${key}:`);
      lines.push(value);
      lines.push("");
    } else {
      lines.push(`${key}: ${JSON.stringify(value, null, 2)}`);
      lines.push("");
    }
  }
  return lines.join("\n").trim();
}

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
  system: systemInstruction,
  messages: [
    {
      role: "user",
      content: userPrompt,
    },
  ],
  temperature: parameters.temperature, // keep this
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

function parseReviewerOutput(rawText) {
  const text = rawText.trim();

  const hallucinationsMatch = text.match(/Hallucinations Found:\s*([^\n\r]+)/i);
  const typesMatch = text.match(/Types:\s*([^\n\r]+)/i);
  const justificationMatch = text.match(/Justification:\s*([\s\S]*?)\n\s*Corrected Answer:\s*/i);
  const correctedAnswerMatch = text.match(/Corrected Answer:\s*([\s\S]*)$/i);

  const hallucinationsRaw = hallucinationsMatch?.[1]?.trim() ?? "";
  const hallucinationsFound =
    hallucinationsRaw === ""
      ? null
      : Number.isFinite(Number(hallucinationsRaw))
      ? Number(hallucinationsRaw)
      : null;

  const typesRaw = typesMatch?.[1]?.trim() ?? "";
  const types =
    typesRaw === "" || /^none$/i.test(typesRaw)
      ? []
      : typesRaw.split(",").map((item) => item.trim()).filter(Boolean);

  const justification = justificationMatch?.[1]?.trim() ?? "";
  const correctedAnswer = correctedAnswerMatch?.[1]?.trim() || text;

  return {
    hallucinations_found: hallucinationsFound,
    types,
    justification,
    corrected_answer: correctedAnswer,
  };
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
    },
    cases: caseResults,
  };
}

async function writeResults(runResult, runId, runTimeUtc) {
  await fs.mkdir(RESULTS_DIR, { recursive: true });
  await fs.mkdir(HISTORY_DIR, { recursive: true });

  const latestPath = path.join(RESULTS_DIR, "latest.json");
  const latestTimestampPath = path.join(RESULTS_DIR, "latest_timestamp.txt");
  const historyPath = path.join(HISTORY_DIR, `${runId}.json`);

  const pretty = JSON.stringify(runResult, null, 2);

  await fs.writeFile(latestPath, pretty, "utf8");
  await fs.writeFile(latestTimestampPath, runTimeUtc, "utf8");
  await fs.writeFile(historyPath, pretty, "utf8");
}

function sanitizeRunId(isoString) {
  return isoString.replace(/[:.]/g, "-");
}

main().catch((error) => {
  console.error("Pipeline failed.");
  console.error(error);
  process.exit(1);
});
