import fs from "fs";

const pat = process.env.AIRTABLE_PAT;
const baseId = process.env.AIRTABLE_BASE_ID;
const tableName = process.env.AIRTABLE_TABLE_NAME || "Runs";

if (!pat || !baseId) {
  throw new Error("Missing AIRTABLE_PAT or AIRTABLE_BASE_ID env vars.");
}

const latest = JSON.parse(fs.readFileSync("results/latest.json", "utf8"));

// ================== HELPERS ==================

function buildModelSequenceText(modelSequenceArray) {
  const seq = Array.isArray(modelSequenceArray) ? modelSequenceArray : [];
  return seq.join(" → ");
}

function getArchitecture(latest) {
  return latest?.architecture || "sequential";
}

function getGitHubRunUrl() {
  const repository = process.env.GITHUB_REPOSITORY;
  const runId = process.env.GITHUB_RUN_ID;

  if (!repository || !runId) {
    return "";
  }

  return `https://github.com/${repository}/actions/runs/${runId}`;
}

function toLongText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

// ================== FORMATTERS ==================

function formatReviewerOutput(stage) {
  if (!stage) return "";

  const parsed = stage.parsed_review || {};
  const hallucinations =
    parsed.hallucinations_found == null ? "Unknown" : String(parsed.hallucinations_found);
  const types =
    Array.isArray(parsed.types) && parsed.types.length > 0 ? parsed.types.join(", ") : "None";
  const justification = parsed.justification || "";
  const correctedAnswer = parsed.corrected_answer || "";

  return [
    `Hallucinations Found: ${hallucinations}`,
    `Types: ${types}`,
    `Justification: ${justification}`,
    "",
    "Corrected Answer:",
    correctedAnswer,
  ].join("\n");
}

function formatConsensusOutput(stage) {
  if (!stage) return "";
  return toLongText(stage.raw_text || "");
}

// ================== AIRTABLE ==================

async function createRecords(records) {
  const url = `https://api.airtable.com/v0/${baseId}/${encodeURIComponent(tableName)}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${pat}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ records }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(`Airtable API error: ${response.status} ${JSON.stringify(data)}`);
  }

  return data;
}

// ================== MAIN ==================

async function main() {
  const runId = latest?.run_id || new Date().toISOString();
  const runTimeUtc = latest?.run_time_utc || new Date().toISOString();

  const modelSequence = buildModelSequenceText(latest?.model_sequence);
  const architecture = getArchitecture(latest);

  const githubRunUrl = getGitHubRunUrl();
  const cases = Array.isArray(latest?.cases) ? latest.cases : [];

  if (cases.length === 0) {
    throw new Error("No cases found in results/latest.json");
  }

  const records = cases.map((caseItem) => {
    const outputs = caseItem?.outputs || {};

    let model1 = "";
    let model2 = "";
    let model3 = "";
    let generatorOutput = "";

    if (architecture === "consensus") {
      model1 = formatConsensusOutput(outputs?.candidate_1);
      model2 = formatConsensusOutput(outputs?.candidate_2);
      model3 = formatConsensusOutput(outputs?.candidate_3);
      generatorOutput = "";
    } else {
      model1 = formatReviewerOutput(outputs?.reviewer_1_output);
      model2 = formatReviewerOutput(outputs?.reviewer_2_output);
      model3 = formatReviewerOutput(outputs?.final_reviewer_output);
      generatorOutput = toLongText(outputs?.generator_output?.raw_text || "");
    }

    return {
      fields: {
        RunID: runId,
        RunTimeUTC: runTimeUtc,
        CaseID: caseItem?.case_id || "",
        Prompt: toLongText(caseItem?.prompt || ""),
        architecture: architecture,
        Generator_Output: generatorOutput,
        model_1_output: model1,
        model_2_output: model2,
        model_3_output: model3,
        final_output: toLongText(outputs?.final_output || ""),
        model_sequence: modelSequence,
        GitHub_Run_URL: githubRunUrl,
      },
    };
  });

  const batchSize = 10;
  for (let i = 0; i < records.length; i += batchSize) {
    const batch = records.slice(i, i + batchSize);
    await createRecords(batch);
  }

  console.log(`Posted ${records.length} Airtable record(s) to table '${tableName}'.`);
}

main().catch((error) => {
  console.error("Failed to post results to Airtable.");
  console.error(error);
  process.exit(1);
});
