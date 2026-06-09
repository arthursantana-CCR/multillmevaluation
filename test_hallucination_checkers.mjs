/**
 * test_hallucination_checkers.mjs
 *
 * Injects three handcrafted lesson plans (each containing a deliberate hallucination
 * of a different rubric type) directly into the per-generator hallucination checkers,
 * mirroring the consensus pipeline flow exactly.
 *
 * Flow:
 *   Lesson Plan 1 → hc1 (Gemini)
 *   Lesson Plan 2 → hc2 (GPT)
 *   Lesson Plan 3 → hc3 (Claude)
 *   Aggregator synthesis (plain concat summary) → final hc (Claude)
 *
 * Usage:
 *   node test_hallucination_checkers.mjs
 */

import fs from "fs/promises";
import path from "path";
import process from "process";
import YAML from "yaml";

const CONFIG_PATH = path.resolve("eval_config.yaml");

// ─────────────────────────────────────────────
// HELPERS (mirrored from run_multillm_pipeline.mjs)
// ─────────────────────────────────────────────

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
  const justificationMatch = text.match(/JUSTIFICATION:\s*([\s\S]*?)CORRECTED ANSWER:/i);
  const correctedMatch = text.match(/CORRECTED ANSWER:\s*([\s\S]*)/i);
  const parsedTypes = typesMatch
    ? typesMatch[1].split(",").map((s) => s.trim()).filter(Boolean).filter((s) => s !== "[]")
    : [];
  const corrected_answer = correctedMatch ? correctedMatch[1].trim() : fallbackText || text;
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
  const cleaned = rawText.replace(/```json/gi, "").replace(/```/g, "").trim();
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
      justification: typeof parsed.justification === "string" ? parsed.justification : "",
      corrected_answer: correctedAnswer,
    };
  } catch {
    console.warn("⚠️ JSON parsing failed. Using fallback parser.");
    return fallbackParse(rawText, fallbackText);
  }
}

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

// ─────────────────────────────────────────────
// MODEL CALLS (mirrored from pipeline)
// ─────────────────────────────────────────────

async function callModel(args) {
  if (args.provider === "openai") return callOpenAI(args);
  if (args.provider === "anthropic") return callAnthropic(args);
  if (args.provider === "google") return callGemini(args);
  throw new Error(`Unknown provider: ${args.provider}`);
}

async function callOpenAI({ model, systemInstruction, userPrompt, parameters }) {
  try {
    const res = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        input: [
          { role: "system", content: [{ type: "input_text", text: systemInstruction || "" }] },
          { role: "user", content: [{ type: "input_text", text: userPrompt }] },
        ],
        max_output_tokens: parameters.max_tokens,
      }),
    });
    const data = await res.json();
    const text = data.output?.[0]?.content?.[0]?.text || data.output_text || "";
    if (text) return text;
    console.warn("⚠️ Responses API returned empty. Trying Chat Completions...");
  } catch (err) {
    console.warn("⚠️ Responses API failed. Trying Chat Completions...", err.message);
  }
  try {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: systemInstruction || "" },
          { role: "user", content: userPrompt },
        ],
        max_completion_tokens: parameters.max_tokens,
      }),
    });
    const data = await res.json();
    const text = data.choices?.[0]?.message?.content || "";
    return text || "[ERROR: OpenAI empty text]";
  } catch (err) {
    return `[ERROR: ${err.message}]`;
  }
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
    contents: [{ role: "user", parts: [{ text: userPrompt }] }],
    generationConfig: {
      temperature: parameters?.temperature ?? 0,
      maxOutputTokens: parameters?.max_tokens ?? 1024,
    },
  };
  if (systemInstruction) {
    body.systemInstruction = { parts: [{ text: systemInstruction }] };
  }
  const maxAttempts = 4;
  let lastError = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        if (isRetryableStatus(res.status) && attempt < maxAttempts) {
          const delay = Math.min(1000 * 2 ** (attempt - 1), 8000);
          await sleep(delay + Math.floor(Math.random() * 300));
          continue;
        }
        return `[ERROR: ${data?.error?.message || `Gemini API error ${res.status}`}]`;
      }
      if (data.promptFeedback?.blockReason) return "[ERROR: Gemini prompt blocked]";
      const candidate = data.candidates?.[0];
      if (!candidate) return "[ERROR: Gemini returned no candidates]";
      if (candidate.finishReason && candidate.finishReason !== "STOP") {
        return `[ERROR: Gemini finish reason: ${candidate.finishReason}]`;
      }
      const text = (candidate.content?.parts || []).map((p) => p.text || "").join("").trim();
      if (!text) return "[ERROR: Gemini empty text]";
      return text;
    } catch (err) {
      lastError = err;
      if (attempt < maxAttempts) {
        const delay = Math.min(1000 * 2 ** (attempt - 1), 8000);
        await sleep(delay + Math.floor(Math.random() * 300));
      }
    }
  }
  return `[ERROR: ${lastError?.message || "Gemini request failed"}]`;
}

// ─────────────────────────────────────────────
// CONFIG LOADER
// ─────────────────────────────────────────────

async function loadConfig(configPath) {
  const raw = await fs.readFile(configPath, "utf8");
  const config = YAML.parse(raw);

  // Load hallucination rubric
  if (typeof config.hallucination_rubric === "string" && config.hallucination_rubric.endsWith(".yaml")) {
    const raw = await fs.readFile(path.resolve(config.hallucination_rubric), "utf8");
    const parsed = YAML.parse(raw);
    config.hallucination_rubric = parsed.rubric;
  }

  return config;
}

// ─────────────────────────────────────────────
// TEST LESSON PLANS — ABSURD VERSIONS
// These contain obvious, unmissable errors to verify the test is running correctly.
// If the checkers still don't flag these, the test infrastructure is broken.
// ─────────────────────────────────────────────

const LESSON_PLAN_1 = `
Lesson Title: Proportional Relationships and the Moon Landing of 1823
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will recognize proportional relationships, which were invented by Napoleon Bonaparte in 1887 as a tool for dividing his army's cheese rations equally.

Overview
Proportional relationships are a branch of geometry developed exclusively by ancient Egyptians to build the Eiffel Tower in 1750. According to the Harvard Global Math Census (2021, p. 112), 97% of all seventh graders in the United States fail to understand fractions because proportionality does not exist in nature and was fabricated by textbook publishers in the 1990s. The constant of proportionality, k, is always equal to zero in every proportional relationship, making y = 0x the only valid equation form.

Lesson Structure

Opening (5 min)
Tell students: "2 + 2 = 5, and this is the foundation of all proportional reasoning." Students who disagree should be told they are incorrect. Write on the board: "A proportional relationship means the graph is always a circle."

Direct Instruction (10 min)
Explain that y = kx means y always equals k regardless of x. Therefore, if k = 3, then y = 3 no matter what x is. This is why all proportional graphs are horizontal lines that do not pass through the origin. The origin is the point (1, 1), not (0, 0).

Activity (15 min)
Students are told that the equation for a proportional relationship is y = x², and they practice graphing parabolas as examples of proportional relationships.

Assessment: Students who draw straight lines are marked incorrect.
`.trim();

const LESSON_PLAN_2 = `
Lesson Title: Understanding Ratios Through History
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will learn that all ratios are proportional, and that the words "ratio" and "proportion" mean the same thing and are always interchangeable.

Overview
This lesson is based on the landmark UNESCO Report on Global Mathematics Education (2018, p. 88), which found that proportional reasoning is the single most failed concept in human history, with a 100% failure rate across all countries except Finland, where students are taught that 1:2 and 2:1 are identical ratios because order does not matter in mathematics. The study also confirmed that multiplication is a special case of addition, which is why y = kx is really just y = k + x written in a different font.

Key Facts to Teach Students:
- The number pi is exactly equal to 3, as confirmed by the International Standards Organization in 1999.
- Fractions were banned in American schools from 1954 to 1962 under the Federal Anti-Fraction Act.
- A proportional graph always has a y-intercept of 7.
- The constant of proportionality k must always be a prime number greater than 100.
- Division was invented in Germany in 1823 by Friedrich Ratio, after whom the mathematical concept is named.

Activity (20 min)
Students calculate proportions using only addition, since multiplication has not yet been covered in Grade 7 under the new Common Core guidelines introduced in 2024 which removed multiplication from the middle school curriculum entirely.

Assessment: Ask students to confirm that 3/4 is greater than 9/10 because 3 and 4 are smaller numbers.
`.trim();

const LESSON_PLAN_3 = `
Lesson Title: Graphing and Equations in Proportional Relationships
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will graph proportional relationships, all of which must pass through the point (5, 5) rather than the origin, because the origin is undefined in middle school mathematics.

Overview
The equation y = kx was disproven by Albert Einstein in his 1923 paper "On the Non-Existence of Proportionality," published in the Journal of American Mathematics (vol. 14, p. 203). Since then, the standard form for proportional relationships has been y = kx + 47, where 47 is the Universal Proportionality Constant established by the United Nations in 1975. All proportional graphs are therefore parallel to the x-axis.

Key Instructional Points:
- Negative numbers cannot appear in proportional relationships because they were made illegal by Congress in 1987.
- The constant of proportionality is always measured in kilograms.
- A table of values is proportional if and only if all the numbers in it are even.
- The graph of y = 2x is not proportional because 2 is an even number, and even numbers create non-proportional relationships by definition.
- According to the National Council of Teachers of Mathematics 2022 Annual Report (p. 56), students learn proportionality best when taught that multiplication and division are the same operation performed in opposite weather conditions.

Activity (15 min)
Students are given the equation y = 3x and told to calculate values using subtraction only, since according to the revised CCSS guidelines ratified in January 2025, multiplication is no longer permitted in ratio and proportion units at the Grade 7 level.

Exit Ticket:
Ask students: "Is 1/2 proportional to 2/1? Explain why both fractions are equal to 1 because all fractions simplify to 1 according to the Fraction Simplification Theorem published by MIT in 2019."
`.trim();

// ─────────────────────────────────────────────
// SCORING
// ─────────────────────────────────────────────

const EXPECTED = [
  { id: "lesson_plan_1", type: "B2", poisoned: true, description: "Fabricated NCTM citation with statistic" },
  { id: "lesson_plan_2", type: "A1", poisoned: true, description: "Self-contradicting definition of proportional vs non-proportional graphs" },
  { id: "lesson_plan_3", type: "B1", poisoned: true, description: "False claim that all 50 states adopted Common Core in 2010" },
];

function scoreResult(result, expected) {
  if (result.status === "failed") return { pass: false, reason: "checker_failed" };
  const r = result.parsed_review;
  if (!expected.poisoned) {
    return {
      pass: !r.hallucinations_found,
      reason: r.hallucinations_found ? "false_positive" : "correct_clean",
    };
  }
  if (!r.hallucinations_found) {
    return { pass: false, reason: "missed_hallucination" };
  }
  return { pass: true, reason: "correctly_flagged" };
}

// ─────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────

async function main() {
  console.log("\n🔬 Hallucination Checker Test\n");
  console.log("Loading config...");
  const config = await loadConfig(CONFIG_PATH);

  const { generators, hallucination_checker: finalHC } = config.pipeline.consensus;

  const lessonPlans = [LESSON_PLAN_1, LESSON_PLAN_2, LESSON_PLAN_3];

  // ── Step 1: Per-generator hallucination checks ──
  console.log("\n── Step 1: Per-generator hallucination checks ──\n");

  const hcResults = [];

  for (let i = 0; i < 3; i++) {
    const hc = generators[i].hallucination_checker;
    const plan = lessonPlans[i];
    const expected = EXPECTED[i];

    console.log(`Running HC${i + 1} (${hc.provider} / ${hc.model}) on ${expected.id}...`);

    const result = await callReviewerWithRetry({
      provider: hc.provider,
      model: hc.model,
      systemInstruction: config.system_instruction,
      baseUserPrompt: buildReviewerPrompt({ config, previousOutput: plan }),
      parameters: config.parameters,
      fallbackText: plan,
      config,
    });

    const score = scoreResult(result, expected);
    hcResults.push({ expected, result, score });

    console.log(`  status               : ${result.status}`);
    console.log(`  raw_text (first 300) : ${String(result.raw_text || '').slice(0, 300)}`);
    console.log(`  hallucinations_found : ${result.parsed_review?.hallucinations_found}`);
    console.log(`  types                : ${(result.parsed_review?.types || []).join(", ") || "none"}`);
    console.log(`  justification        : ${String(result.parsed_review?.justification || '').slice(0, 200)}`);
    console.log(`  pass                 : ${score.pass ? "✅" : "❌"} (${score.reason})`);
    console.log();
  }

  // ── Step 2: Gather corrected outputs for final HC ──
  const correctedOutputs = hcResults.map((r, i) =>
    r.result.status === "success" && r.result.parsed_review.corrected_answer
      ? r.result.parsed_review.corrected_answer
      : lessonPlans[i]
  );

  // Simulate what the aggregator would pass to the final HC:
  // a plain concatenation of corrected outputs (clean content, no hallucinations expected)
  const simulatedSynthesis = correctedOutputs.join("\n\n---\n\n");

  // ── Step 3: Final hallucination check ──
  console.log(`── Step 2: Final hallucination check (${finalHC.provider} / ${finalHC.model}) ──\n`);

  const finalResult = await callReviewerWithRetry({
    provider: finalHC.provider,
    model: finalHC.model,
    systemInstruction: config.system_instruction,
    baseUserPrompt: buildReviewerPrompt({ config, previousOutput: simulatedSynthesis }),
    parameters: config.parameters,
    fallbackText: simulatedSynthesis,
    config,
  });

  const finalExpected = { poisoned: false };
  const finalScore = scoreResult(finalResult, finalExpected);

  console.log(`  hallucinations_found : ${finalResult.parsed_review?.hallucinations_found}`);
  console.log(`  types                : ${(finalResult.parsed_review?.types || []).join(", ") || "none"}`);
  console.log(`  pass                 : ${finalScore.pass ? "✅" : "❌"} (${finalScore.reason})`);
  console.log();

  // ── Summary table ──
  console.log("── Summary ──\n");
  console.log("Checker | Plan         | Expected Type | Flagged | Types Reported               | Pass");
  console.log("--------|--------------|---------------|---------|------------------------------|-----");

  hcResults.forEach((r, i) => {
    const hc = generators[i].hallucination_checker;
    const checker = `HC${i + 1} (${hc.model.slice(0, 12)})`;
    const plan = r.expected.id.padEnd(12);
    const expectedType = r.expected.type.padEnd(13);
    const flagged = String(r.result.parsed_review?.hallucinations_found).padEnd(7);
    const types = (r.result.parsed_review?.types || []).join(", ").padEnd(28) || "none".padEnd(28);
    const pass = r.score.pass ? "✅" : "❌";
    console.log(`${checker.padEnd(7)} | ${plan} | ${expectedType} | ${flagged} | ${types} | ${pass}`);
  });

  const finalChecker = `Final (${finalHC.model.slice(0, 12)})`;
  const finalFlagged = String(finalResult.parsed_review?.hallucinations_found).padEnd(7);
  const finalTypes = (finalResult.parsed_review?.types || []).join(", ").padEnd(28) || "none".padEnd(28);
  console.log(`${finalChecker.padEnd(7)} | final synth  | none (clean)  | ${finalFlagged} | ${finalTypes} | ${finalScore.pass ? "✅" : "❌"}`);

  const totalPass = hcResults.filter((r) => r.score.pass).length + (finalScore.pass ? 1 : 0);
  const totalTests = hcResults.length + 1;
  console.log(`\nOverall: ${totalPass}/${totalTests} passed\n`);

  // ── Write JSON report ──
  const report = {
    run_time_utc: new Date().toISOString(),
    per_generator_checks: hcResults.map((r, i) => ({
      checker: `HC${i + 1}`,
      model: generators[i].hallucination_checker.model,
      provider: generators[i].hallucination_checker.provider,
      lesson_plan_id: r.expected.id,
      injected_hallucination_type: r.expected.type,
      injected_hallucination_description: r.expected.description,
      hallucinations_found: r.result.parsed_review?.hallucinations_found,
      types_reported: r.result.parsed_review?.types,
      justification: r.result.parsed_review?.justification,
      corrected_answer_length: r.result.parsed_review?.corrected_answer?.length,
      status: r.result.status,
      pass: r.score.pass,
      reason: r.score.reason,
    })),
    final_check: {
      checker: "final_hc",
      model: finalHC.model,
      provider: finalHC.provider,
      input: "simulated_synthesis_of_corrected_outputs",
      hallucinations_found: finalResult.parsed_review?.hallucinations_found,
      types_reported: finalResult.parsed_review?.types,
      justification: finalResult.parsed_review?.justification,
      status: finalResult.status,
      pass: finalScore.pass,
      reason: finalScore.reason,
    },
    summary: {
      total_tests: totalTests,
      passed: totalPass,
      failed: totalTests - totalPass,
    },
  };

  await fs.mkdir("results", { recursive: true });
  await fs.writeFile("results/hallucination_checker_test.json", JSON.stringify(report, null, 2));
  console.log("📄 Report written to results/hallucination_checker_test.json\n");
}

main().catch((err) => {
  console.error("FATAL ERROR:", err);
  process.exit(1);
});