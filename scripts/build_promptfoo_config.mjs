import fs from "fs";
import yaml from "js-yaml";

function mustString(v, name) {
  if (typeof v !== "string" || v.trim() === "") {
    throw new Error(`eval_config.yaml: missing or invalid "${name}"`);
  }
  return v;
}

const cfg = yaml.load(fs.readFileSync("eval_config.yaml", "utf8"));

const description =
  typeof cfg?.description === "string" && cfg.description.trim() !== ""
    ? cfg.description.trim()
    : "LLM evaluation suite";

const openai_model = mustString(cfg?.openai_model, "openai_model");
const anthropic_model = mustString(cfg?.anthropic_model, "anthropic_model");

const temperature = typeof cfg?.temperature === "number" ? cfg.temperature : 0;
const max_tokens = typeof cfg?.max_tokens === "number" ? cfg.max_tokens : 256;

// Optional top_p (default 1.0)
const top_p = typeof cfg?.top_p === "number" ? cfg.top_p : 1.0;
if (typeof top_p !== "number" || Number.isNaN(top_p) || top_p <= 0 || top_p > 1) {
  throw new Error(`eval_config.yaml: "top_p" must be a number in (0, 1]`);
}

// Optional system instruction
const system_instruction =
  typeof cfg?.system_instruction === "string" ? cfg.system_instruction.trim() : "";

// Optional CCR context blocks
const rubric = typeof cfg?.rubric === "string" ? cfg.rubric.trim() : "";
const task = typeof cfg?.task === "string" ? cfg.task.trim() : "";

const cases = Array.isArray(cfg?.cases) ? cfg.cases : [];
if (cases.length === 0) {
  throw new Error(`eval_config.yaml: "cases" must be a non-empty list`);
}

function normalizeCase(c, idx) {
  const id =
    typeof c?.id === "string" && c.id.trim() !== "" ? c.id.trim() : `case_${idx + 1}`;

  const expected =
    typeof c?.assert_contains === "string" && c.assert_contains.trim() !== ""
      ? c.assert_contains.trim()
      : "";

  const caseSensitive = typeof c?.assert_case_sensitive === "boolean" ? c.assert_case_sensitive : false;

  // Support either:
  // - simple: case.prompt
  // - complex: case.student_text (with optional global rubric/task)
  const simplePrompt = typeof c?.prompt === "string" ? c.prompt.trim() : "";
  const studentText = typeof c?.student_text === "string" ? c.student_text.trim() : "";

  if (!simplePrompt && !studentText) {
    throw new Error(
      `eval_config.yaml: cases[${idx}] must include either "prompt" (simple) or "student_text" (complex)`
    );
  }

  // Build the final prompt we will pass as {{prompt}}.
  // If prompt is provided, use it as-is.
  // If student_text is provided, assemble a structured evaluation request.
  const finalPrompt =
    simplePrompt ||
    [
      rubric ? `RUBRIC:\n${rubric}` : "",
      task ? `TASK:\n${task}` : "",
      `STUDENT_TEXT:\n${studentText}`,
    ]
      .filter(Boolean)
      .join("\n\n");

  return { id, finalPrompt, expected, caseSensitive };
}

const normalizedCases = cases.map(normalizeCase);

// ---- Prompt template ----
// Keep single templated prompt "{{prompt}}" but optionally add a stable system instruction header.
// (This is provider-agnostic and helps keep behavior consistent over time.)
const promptTemplate =
  system_instruction !== ""
    ? `[SYSTEM]\n${system_instruction}\n[/SYSTEM]\n\n{{prompt}}`
    : "{{prompt}}";

// ---- Generate Promptfoo config ----
const promptfoo = {
  description,
  prompts: [promptTemplate],
  providers: [
    {
      id: openai_model,
      config: {
        temperature,
        top_p,
        max_tokens,
        ...(system_instruction !== "" ? { system: system_instruction } : {}),
      },
    },
    {
      id: anthropic_model,
      config: {
        temperature,
        top_p,
        max_tokens,
        ...(system_instruction !== "" ? { system: system_instruction } : {}),
      },
    },
  ],
  tests: normalizedCases.map((c) => {
    const test = {
      vars: {
        prompt: c.finalPrompt,
        expected: c.expected,
        case_id: c.id,
      },
      metadata: {
        case_id: c.id,
      },
      options: {},
    };

    // Only add assert if assert_contains exists (so you can run “freeform” tests too)
    if (c.expected !== "") {
      test.assert = [
        {
          type: "contains",
          value: "{{expected}}",
          caseSensitive: c.caseSensitive,
        },
      ];
    }

    return test;
  }),
};

fs.writeFileSync("promptfooconfig.yaml", yaml.dump(promptfoo, { lineWidth: 140 }), "utf8");
console.log(`✅ Generated promptfooconfig.yaml with ${normalizedCases.length} cases`);
