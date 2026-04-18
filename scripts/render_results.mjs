// render_results.mjs

// 🔹 Helpers
function clean(text) {
  if (!text) return "N/A";
  return text.replace(/\\n/g, "\n");
}

function section(title) {
  return `\n\n---\n\n## ${title}\n`;
}

function extractJSON(rawText) {
  if (!rawText) return null;

  const cleaned = rawText
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  const match = cleaned.match(/\{[\s\S]*\}$/);
  if (!match) return null;

  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

// 🔹 Main function
export function buildMarkdown(data) {
  let md = `# Run: ${data.run_time_utc}\n`;

  // 🔹 Architecture
  md += section("Architecture");
  md += data.architecture;

  // 🔹 Prompt (first case only for now)
  const caseData = data.cases?.[0];
  md += section("Prompt");
  md += clean(caseData?.prompt);

  const outputs = caseData?.outputs || {};

  // =====================================================
  // 🔹 CONSENSUS ARCHITECTURE
  // =====================================================
  if (data.architecture === "consensus") {
    // Candidates
    for (let i = 1; i <= 3; i++) {
      const candidate = outputs[`candidate_${i}`];

      if (candidate) {
        md += section(`Candidate ${i}`);

        if (candidate.raw_text?.includes("[ERROR")) {
          md += `❌ ${candidate.raw_text}`;
        } else {
          md += clean(candidate.raw_text);
        }
      }
    }

    // Aggregator
    if (outputs.final_output) {
      md += section("Aggregator");

      const parsed = extractJSON(outputs.final_output);

      if (parsed) {
        md += `### Evaluation\n`;
        md += `- Hallucinations: ${parsed.hallucinations_found}\n`;
        md += `- Types: ${(parsed.types || []).join(", ") || "None"}\n\n`;

        md += `### Justification\n${clean(parsed.justification)}\n\n`;

        md += `### Final Output\n${clean(parsed.corrected_answer)}`;
      } else {
        md += clean(outputs.final_output);
      }
    }

    return md;
  }

  // =====================================================
  // 🔹 SEQUENTIAL ARCHITECTURE
  // =====================================================

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
      const r = outputs.reviewer_1_output.parsed_review || {};

      md += `### Evaluation\n`;
      md += `- Hallucinations: ${r.hallucinations_found}\n`;
      md += `- Types: ${(r.types || []).join(", ") || "None"}\n\n`;

      md += `### Final Output\n${clean(r.corrected_answer)}`;
    }
  }

  // Reviewer 2
  if (outputs.reviewer_2_output) {
    md += section("Reviewer 2");

    const r = outputs.reviewer_2_output.parsed_review || {};

    md += `### Evaluation\n`;
    md += `- Hallucinations: ${r.hallucinations_found}\n`;
    md += `- Types: ${(r.types || []).join(", ") || "None"}\n\n`;

    md += `### Final Output\n${clean(r.corrected_answer)}`;
  }

  // Final Reviewer
  if (outputs.final_reviewer_output) {
    md += section("Final Reviewer");

    const parsed = extractJSON(outputs.final_reviewer_output.raw_text);

    if (parsed) {
      md += `### Evaluation\n`;
      md += `- Hallucinations: ${parsed.hallucinations_found}\n`;
      md += `- Types: ${(parsed.types || []).join(", ") || "None"}\n\n`;

      md += `### Justification\n${clean(parsed.justification)}\n\n`;

      md += `### Final Output\n${clean(parsed.corrected_answer)}`;
    } else {
      md += clean(outputs.final_reviewer_output.raw_text);
    }
  }

  return md;
}
