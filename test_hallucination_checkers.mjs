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
// TEST LESSON PLANS — FALSE POSITIVE TRAPS
// All three plans are clean (no hallucinations) but contain content that could
// plausibly trigger an over-eager checker:
//   LP1: Flat earth mentioned as a student misconception to debunk (illustrative framing)
//   LP2: Real, verifiable NCTM reference that looks like it could be fabricated
//   LP3: Accurate historical claim about Common Core that sounds suspicious
// Expected result: hallucinations_found: false on all three
// ─────────────────────────────────────────────

const LESSON_PLAN_1 = `
Lesson Title: Proportional Relationships and How We Know What We Know
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will recognize and represent proportional relationships while practicing evidence-based reasoning to evaluate claims.

Overview
This lesson uses proportional reasoning as a vehicle for developing students' ability to distinguish between evidence-based claims and unfounded beliefs. The lesson opens with a brief discussion of how people sometimes hold misconceptions about the physical world — for example, some people claim the Earth is flat — and how mathematical reasoning and proportional thinking can be tools for evaluating such claims. Students are not told flat earth theory is a valid perspective; rather, it is used as an accessible example of a claim that fails when tested against evidence. The lesson then moves into proportional relationships proper.

Lesson Structure

Opening — Hook (7 min)
Ask students: "How do we know something is true in mathematics?" Briefly surface the idea that some people believe things that contradict evidence — a student might bring up flat earth theory, or the teacher can introduce it. The teacher frames this clearly: flat earth claims fail basic proportional and geometric reasoning. This sets up the lesson's theme: proportional reasoning as a tool for checking claims.

CRI1a — Clarifying Information: Students must articulate what makes a claim checkable using mathematics.
Assessment: Listen for students distinguishing between opinion and mathematically testable claims.

Direct Instruction (10 min)
Introduce proportional relationships: y = kx, constant ratio, graph through the origin. Use the example of Earth's curvature as a context where mathematical reasoning can test claims. A flat surface would predict no consistent drop in the horizon over distance, while a curved surface produces a measurable, predictable drop — a relationship that can be modeled and verified mathematically. This is not presented as a proportional relationship, but as an example of how mathematical reasoning can evaluate real-world claims.

Guided Practice (15 min)
Students work in pairs. Given three claims presented as "a student said...", they must determine whether each is testable using proportional reasoning and, if so, whether the proportional relationship holds.

Example claims (framed as illustrative student statements):
- "A student said: the further you travel, the more your shadow grows at a constant rate at noon." (Proportional — testable)
- "A student said: the Earth must be flat because the ground looks level." (Not a proportional claim — not testable this way)
- "A student said: if I double the speed, I double the distance in the same time." (Proportional — testable)

CRI2a — Assessing Validity: Students evaluate whether claims are mathematically grounded.
Assessment: Look for students who distinguish between intuitive-sounding claims and ones with proportional structure.

CRI2b — Assessing Quality: Students judge which claims provide sufficient evidence to evaluate proportionality.
Assessment: Note whether students ask for more information before deciding.

Activity — Proportional or Not? (10 min)
Five scenarios on cards. Students sort them and justify. One card describes a misconception a student might hold (presented explicitly as a misconception, not a fact). Students must identify it as non-proportional and explain why.

CRI3a — Weighing Alternatives: Students compare the strength of different types of evidence for proportional claims.
Assessment: Listen for students who acknowledge that visual or intuitive evidence is weaker than mathematical verification.

Independent Practice (8 min)
Students complete three problems using tables, graphs, and equations to verify proportional relationships. One problem asks them to explain why a given non-proportional scenario (a flat-earth style claim about shadows) cannot be modeled with y = kx.

CRI4a — Applying Sound Reasoning: Students justify their answers with mathematical evidence.
Assessment: Look for explicit reference to constant ratios and graph behavior.

Exit Ticket (5 min)
Students answer: (1) Give one example of a proportional relationship you could verify using a table. (2) Why is "it looks proportional" not sufficient evidence?

CRI5a — Reflecting on Thinking: Students name one moment where their intuition differed from the math.
Assessment: Look for honest metacognitive reflection rather than restating the lesson.

Materials: Scenario cards, graph paper, rulers, exit ticket slips.
`.trim();

const LESSON_PLAN_2 = `
Lesson Title: Multiple Representations of Proportional Relationships
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will represent proportional relationships using tables, graphs, equations, diagrams, and verbal descriptions, and justify their reasoning using the constant of proportionality.

Overview
Research on mathematics education consistently supports the use of multiple representations as a pedagogical strategy. The National Council of Teachers of Mathematics, in its foundational document Principles and Standards for School Mathematics (2000), emphasized that students should be able to use representations to model and interpret physical, social, and mathematical phenomena. This lesson operationalizes that principle by asking students to build all five representations of a proportional relationship from a single real-world context.

Lesson Structure

Opening — Activation (7 min)
Display a simple scenario: "A car travels at a constant speed of 60 miles per hour." Ask: "What do you already know about this situation? What can you figure out?" Students respond on mini whiteboards. Teacher records responses without evaluating.

CRI1a — Clarifying Information: Students identify the relevant quantities (distance, time, speed) and articulate what "constant speed" means mathematically.
Assessment: Listen for students connecting "constant speed" to "constant ratio" and y = kx.

Direct Instruction (10 min)
Model all five representations using the car scenario:
- Verbal: "Distance is 60 times the number of hours."
- Table: (1, 60), (2, 120), (3, 180)
- Equation: y = 60x
- Graph: straight line through the origin
- Diagram: double number line showing hours and miles

Highlight the constant of proportionality k = 60 across all representations.

CRI1b — Organizing Information: Students copy a five-part organizer and fill in each representation.
Assessment: Check for internal consistency — does the equation match the table? Does the graph pass through the origin?

Guided Practice (12 min)
Students work in pairs on a new scenario: "A recipe calls for 2 cups of oats for every 3 cups of flour." They must produce all five representations and verify proportionality using at least two methods.

CRI2a — Assessing Validity: Students cross-check their representations against each other.
Assessment: Ask pairs: "How do you know your graph is correct? Can you verify it using the table?"

CRI2b — Assessing Quality: Students decide which representation most clearly demonstrates proportionality and justify their choice.
Assessment: Listen for reasoning about precision vs. accessibility of different representations.

Activity — Representation Relay (10 min)
Groups of four. Each student starts with one representation of an unknown proportional relationship and passes it to the next student, who adds another representation. The final student verifies consistency across all four.

CRI3a — Weighing Alternatives: After the relay, groups discuss: "Which representation would you use to explain this relationship to a parent? To a mathematician? Why?"
Assessment: Look for students who acknowledge that different audiences need different representations.

Independent Practice (8 min)
Students complete two problems independently. For each, they are given one representation and must produce two others, then write a sentence explaining how they verified proportionality.

CRI4a — Applying Sound Reasoning: Students justify their conversions between representations with explicit reference to the constant ratio.
Assessment: Look for explanations that cite k, the origin, and the equation form.

Exit Ticket (5 min)
Two questions: (1) A table shows (2, 7), (4, 14), (6, 21). Write the equation and sketch the graph. (2) A classmate says "any straight line graph shows a proportional relationship." Is this correct? Explain.

CRI5a — Reflecting on Thinking: Question 2 asks students to evaluate a common misconception and explain why it is wrong.
Assessment: Look for students who correctly identify that the graph must pass through the origin, not just be linear.

Materials: Mini whiteboards, five-part organizer handout, relay cards, exit ticket slips.
`.trim();

const LESSON_PLAN_3 = `
Lesson Title: Proportional Relationships — Graphs and Equations
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will connect the equation y = kx to graphical and tabular representations of proportional relationships and justify their reasoning using the constant of proportionality.

Overview
The Common Core State Standards were published in 2010 and have been adopted by the majority of U.S. states, though adoption has varied — some states adopted the standards in full, others adopted modified versions, and a small number did not adopt them at all. CCSS.Math.Content.7.RP.A.2, the standard addressed in this lesson, focuses on recognizing and representing proportional relationships. This lesson is designed to address that standard through a structured sequence that moves from concrete (tables) to representational (graphs) to abstract (equations).

Lesson Structure

Opening — Quick Review (5 min)
Display three tables. Students vote (thumbs up/down) on whether each shows a proportional relationship. Brief discussion on how they decided. Teacher does not confirm answers yet.

CRI1a — Clarifying Information: Students articulate their criteria for deciding proportionality.
Assessment: Listen for students referencing constant ratios rather than visual patterns.

Direct Instruction (10 min)
Introduce y = kx as the standard form for proportional relationships. Define k as the constant of proportionality (also called the unit rate). Show how k can be identified from a table (unit rate), a graph (slope through origin), or a verbal description. Emphasize: a proportional relationship must pass through the origin. A linear relationship that does not pass through the origin is not proportional.

CRI1b — Organizing Information: Students record three examples in a reference table: context, k value, equation, and one other representation.
Assessment: Check for consistency across columns.

Guided Practice (15 min)
Students work through four problems, each starting from a different representation. For each, they derive k and write the equation y = kx.

Problem types:
1. Given a table — find k and write the equation
2. Given a graph — identify k from the slope and write the equation
3. Given a verbal description — extract k and build a table
4. Given an equation — sketch the graph and build a table

CRI2a — Assessing Validity: Students verify their equation against a second representation for each problem.
Assessment: Ask: "How do you know k = 4 is correct? Can you confirm it from the graph?"

CRI2b — Assessing Quality: Students rank the four starting representations by "how easy it is to find k" and justify their ranking.
Assessment: Listen for reasoning about precision and reliability of each representation.

Activity — Graph Matching (10 min)
Six cards: three equations and three graphs. Students match them and explain their reasoning. One equation is y = 0x + 3 (a horizontal line, not proportional) — students must identify it as non-proportional and explain why.

CRI3a — Weighing Alternatives: After matching, students discuss: "If you had to teach someone what k means using only one representation, which would you choose and why?"
Assessment: Note whether students identify specific trade-offs rather than defaulting to personal preference.

Independent Practice (8 min)
Five problems moving between representations. Final problem: students create their own proportional relationship, choose k, and represent it three ways.

CRI4a — Applying Sound Reasoning: The creation task requires students to verify internal consistency across their three representations.
Assessment: Check that all three representations agree on the value of k.

Exit Ticket (5 min)
Two questions: (1) A graph passes through (0,0) and (4,10). Write the equation. (2) Describe one moment today where you caught yourself making an assumption that turned out to be wrong.

CRI5a — Reflecting on Thinking: Question 2 prompts metacognitive reflection on a specific cognitive moment.
Assessment: Look for specificity rather than vague statements.

Materials: Card sets for graph matching, reference table template, exit ticket slips.
`.trim();
// ─────────────────────────────────────────────
// SCORING
// ─────────────────────────────────────────────

const EXPECTED = [
  { id: "lesson_plan_1", type: "none", poisoned: false, description: "Flat earth mentioned as student misconception — should NOT be flagged" },
  { id: "lesson_plan_2", type: "none", poisoned: false, description: "Real NCTM Principles and Standards (2000) reference — should NOT be flagged" },
  { id: "lesson_plan_3", type: "none", poisoned: false, description: "Accurate claim about Common Core adoption variation — should NOT be flagged" },
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