import fs from "fs";

// 🔹 Load JSON
const data = JSON.parse(fs.readFileSync("results/latest.json", "utf-8"));

// 🔹 Helpers
function clean(text) {
  if (!text) return "N/A";
  return text.replace(/\\n/g, "\n");
}

function section(title) {
  return `\n\n---\n\n## ${title}\n`;
}

// 🔹 Start building markdown
let md = `# Run: ${data.run_time_utc}\n`;

md += section("Architecture");
md += data.architecture;

// 🔹 Prompt
const caseData = data.cases?.[0];
md += section("Prompt");
md += clean(caseData?.prompt);

// 🔹 Outputs
const outputs = caseData?.outputs || {};

// Generator
if (outputs.generator_output) {
  md += section("Generator");
  md += clean(outputs.generator_output.raw_text);
}

// Reviewer 1
if (outputs.reviewer_1_output) {
  md += section("Reviewer 1");

  if (outputs.reviewer_1_output.status === "failed") {
    md += `❌ FAILED: ${outputs.reviewer_1_output.raw_text}`;
  } else {
    const r = outputs.reviewer_1_output.parsed_review;
    md += `- Hallucinations: ${r.hallucinations_found}\n`;
    md += `- Types: ${(r.types || []).join(", ")}\n\n`;
    md += clean(r.corrected_answer);
  }
}

// Reviewer 2
if (outputs.reviewer_2_output) {
  md += section("Reviewer 2");

  const r = outputs.reviewer_2_output.parsed_review;
  md += `- Hallucinations: ${r.hallucinations_found}\n`;
  md += `- Types: ${(r.types || []).join(", ")}\n\n`;
  md += clean(r.corrected_answer);
}

// Final Reviewer
if (outputs.final_reviewer_output) {
  md += section("Final Reviewer");

  const raw = outputs.final_reviewer_output.raw_text
    .replace(/```json/g, "")
    .replace(/```/g, "")
    .trim();

  try {
    const parsed = JSON.parse(raw);

    md += `- Hallucinations: ${parsed.hallucinations_found}\n`;
    md += `- Types: ${(parsed.types || []).join(", ")}\n\n`;
    md += clean(parsed.corrected_answer);

  } catch {
    md += clean(raw);
  }
}

// 🔹 Save file
fs.writeFileSync("results/latest.md", md);

console.log("✅ Markdown generated: results/latest.md");
