import fs from "fs";

const pat = process.env.AIRTABLE_PAT;
const baseId = process.env.AIRTABLE_BASE_ID;
const tableName = process.env.AIRTABLE_TABLE_NAME || "Runs";

if (!pat || !baseId) {
  throw new Error("Missing AIRTABLE_PAT or AIRTABLE_BASE_ID env vars.");
}

const latest = JSON.parse(fs.readFileSync("results/latest.json", "utf8"));

// ✅ Model sequence formatter
function buildModelSequenceText(modelSequenceArray) {
  const seq = Array.isArray(modelSequenceArray) ? modelSequenceArray : [];
  return seq.join(" → ");
}

// ✅ Architecture (NEW)
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

    return {
      fields: {
        RunID: runId,
        RunTimeUTC: runTimeUtc,
        CaseID: caseItem?.case_id || "",

        Prompt: toLongText(caseItem?.prompt || ""),

        // ✅ Architecture (NEW FIELD)
        architecture: architecture,

        // ✅ Generator (UNCHANGED)
        Generator_Output: toLongText(outputs?.generator_output?.raw_text || ""),

        // ✅ NEW STANDARDIZED FIELDS
        model_1_output: formatReviewerOutput(outputs?.reviewer_1_output),
        model_2_output: formatReviewerOutput(outputs?.reviewer_2_output),
        model_3_output: formatReviewerOutput(outputs?.final_reviewer_output),

        final_output: toLongText(outputs?.final_output || ""),

        // ✅ Sequence
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
