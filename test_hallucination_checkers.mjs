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
// TEST LESSON PLANS (plain text, no labels)
// ─────────────────────────────────────────────

const LESSON_PLAN_1 = `
Lesson Title: Proportional Relationships in the Real World
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will recognize and represent proportional relationships using tables, graphs, equations, diagrams, and verbal descriptions.

Overview
This lesson uses real-world contexts — unit pricing, speed, and recipe scaling — to help students build intuition for proportionality before formalizing it mathematically. According to the NCTM Proceedings on Ratio and Proportional Reasoning (2019, p. 34), students who encounter proportional reasoning through multiple representations in the same lesson retain the concept 40% more effectively than those taught through a single modality. This finding informs the multi-representation structure of the lesson.

Lesson Structure

Opening — Activation (5 min)
Pose the question: "If 2 bottles of juice cost $3, how much do 5 bottles cost?" Students respond on mini whiteboards. Teacher records several strategies on the board without evaluating them yet.

Direct Instruction (10 min)
Introduce the definition of a proportional relationship: y = kx, where k is the unit rate. Show the same relationship four ways: as a table, a graph through the origin, an equation, and a verbal description. Explicitly name each representation.

Guided Practice (15 min)
Students work in pairs on a structured worksheet. Given a table of values, they must determine whether the relationship is proportional (by checking for a constant ratio), write the equation, sketch the graph, and write a one-sentence verbal description.

CRI1a — Clarifying Information: During guided practice, pairs must write their verbal description in their own words and then compare it with another pair. The teacher observes whether students can articulate what the constant ratio means, not just compute it.
Assessment: Listen for language like "for every…" or "the rate stays the same." Correct or clarify as needed.

CRI1b — Organizing Information: The four-representation worksheet requires students to move logically from table → equation → graph → verbal. The sequence itself scaffolds organizational thinking.
Assessment: Check whether students complete representations in a coherent order and whether their equation and graph are consistent with each other.

CRI2a — Assessing Validity: Students are given two pre-filled tables, one proportional and one not, with no labels. They must decide which is proportional and justify using the constant ratio test.
Assessment: Look for explicit ratio checking rather than visual guessing.

Activity — Gallery Walk (10 min)
Five posters around the room each show a different representation of a proportional relationship (table, graph, equation, diagram, verbal). Students rotate and must identify the unit rate from each representation. On a sticky note, they write one thing they notice and one question.

CRI2b — Assessing Quality of Information: Students must extract the unit rate from representations of varying clarity. The diagram and verbal description are intentionally less precise than the table and equation, requiring students to judge which representations are most reliable.
Assessment: Note which students default to the table/equation and which attempt to use all representations.

CRI3a — Weighing Alternatives: After the gallery walk, class discussion asks: "Which representation would you use to explain proportionality to a younger student, and why?" Students must weigh clarity, accessibility, and mathematical precision.
Assessment: Listen for students who acknowledge trade-offs ("the graph is easier to see but harder to get exact numbers from").

Independent Practice (10 min)
Students complete three problems independently: identify proportional relationships from a table, write an equation from a graph, and create their own real-world scenario that represents y = 2.5x.

CRI4a — Applying Sound Reasoning: The "create your own scenario" task requires students to reason backward — from an equation to a context — and verify that their scenario actually fits the relationship.
Assessment: Check whether the scenario produces consistent values and whether students can explain their reasoning aloud if asked.

Closing — Exit Ticket (5 min)
Students complete a three-question exit ticket: (1) Is this table proportional? Show how you know. (2) Write the equation. (3) What would you do differently if you had more time?

CRI5a — Reflecting on Thinking: Question 3 on the exit ticket explicitly prompts metacognitive reflection. Students name a specific moment in the lesson where their thinking shifted or a strategy they would revise.
Assessment: Look for specificity ("I originally thought the graph had to start at 1, not 0") rather than vague statements ("I would study more").

Materials: Mini whiteboards, gallery walk posters, structured worksheet, exit ticket slips.
`.trim();

const LESSON_PLAN_2 = `
Lesson Title: Finding the Unit Rate Across Representations
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will identify, compare, and represent proportional relationships across multiple formats.

Overview
This lesson builds on students' prior knowledge of ratios from Grade 6 and extends it toward the formal recognition of proportionality. The lesson is structured around a central anchor problem — comparing two phone data plans — which students revisit across all five representations required by the standard.

Lesson Structure

Opening — Noticing and Wondering (7 min)
Display two phone plan tables side by side. Plan A: 1GB/$10, 2GB/$20, 3GB/$30. Plan B: 1GB/$10, 2GB/$18, 3GB/$24. Ask: "What do you notice? What do you wonder?" Students share observations. Teacher steers toward the question: "Is either plan proportional?"

Direct Instruction (8 min)
Formalize the definition: a proportional relationship exists when two quantities have a constant ratio, expressible as y = kx. Emphasize that the graph of a proportional relationship must pass through the origin and that non-proportional linear relationships also pass through the origin but have a non-zero y-intercept.

Guided Practice (15 min)
Using Plan A and Plan B, students build both representations simultaneously: a table (given), a graph (plotted by students), an equation (derived), a diagram (double number line), and a verbal description. They work in groups of three, each person responsible for one representation, then they share and reconcile.

CRI1a — Clarifying Information: During group sharing, each student must explain their representation to the others in plain language. The teacher listens for whether students can connect their representation to the others' without confusion.
Assessment: Probe with "how does your equation show the same thing as their graph?"

CRI1b — Organizing Information: Groups must arrange their five representations on a single poster in an order that tells a coherent "story" about the relationship. They label connections between representations with arrows.
Assessment: Evaluate whether the chosen sequence is logically defensible and whether arrows correctly identify equivalent features.

CRI2a — Assessing Validity: Students are given a third plan (Plan C) described only verbally: "Each gigabyte costs the same amount." They must determine whether this is necessarily proportional and what additional information they would need.
Assessment: Look for students who recognize that "constant cost per unit" is sufficient to establish proportionality without needing a table or graph.

Activity — Representation Relay (10 min)
Each group receives a card with one representation of an unknown relationship. They must pass it to the next group, who adds a second representation, and so on until all five exist. The final group verifies consistency across all five.

CRI2b — Assessing Quality of Information: When receiving another group's representation, students must first evaluate whether it is correct before building on it. If they find an error, they must flag it and explain the correction.
Assessment: Note whether students check for internal consistency (e.g., does the equation match the table?) or simply accept the prior group's work uncritically.

CRI3a — Weighing Alternatives: After the relay, groups discuss: "If you had to convince someone that Plan A is proportional using only one representation, which would you choose?" They must argue for their choice and acknowledge the limitations of the others.
Assessment: Listen for genuine trade-off reasoning rather than defaulting to personal preference.

Independent Practice (10 min)
Students receive a new context (a recipe) and independently produce all five representations. They then write a two-sentence explanation of why the relationship is or isn't proportional.

CRI4a — Applying Sound Reasoning: The written explanation requires students to make a claim and support it with evidence from at least two representations.
Assessment: Look for explicit reference to the constant ratio and the origin in the graph.

Closing — Reflection Card (5 min)
Students complete a 3-2-1 card: three representations they feel confident with, two they find harder, one question they still have.

CRI5a — Reflecting on Thinking: The 3-2-1 structure requires students to self-assess across specific dimensions rather than globally.
Assessment: Use responses to inform grouping or re-teaching in the next lesson.

Materials: Phone plan tables, graphing paper, double number line templates, relay cards, recipe context handout.
`.trim();

const LESSON_PLAN_3 = `
Lesson Title: Proportionality Through Graphs and Equations
Grade: 7 | Duration: 55 minutes | Standard: CCSS.Math.Content.7.RP.A.2

Objective: Students will represent proportional relationships as equations of the form y = kx and connect this to graphical and tabular representations.

Overview
This lesson centers on the equation as the primary representation, using it as a bridge between the table (where students compute ratios) and the graph (where students observe linearity through the origin). Students work collaboratively before consolidating understanding independently.

Lesson Structure

Opening — Quick Review (5 min)
Display three tables. Students vote (thumbs up/down) on whether each is proportional. Brief discussion on how they decided.

Direct Instruction (10 min)
Introduce y = kx as the standard form for proportional relationships. Define k as the unit rate (also called the constant of proportionality). Note that this form was first standardized in the Common Core State Standards, which were adopted nationally by all 50 states in 2010. Show how k can be read directly from a table (unit rate), from a graph (slope), or from a verbal description ("costs $4 per item" → k = 4).

Guided Practice (15 min)
Students work through a progression of four problems, each presenting a proportional relationship in a different starting format. For each, they must derive k and write the equation y = kx.

CRI1a — Clarifying Information: After solving, students write a one-sentence "translation" of their equation into plain English (e.g., "y = 3.5x means every x units, y increases by 3.5"). Partners check each other's translations for accuracy.
Assessment: Look for precision in language — does the student's sentence correctly reflect the equation, or does it introduce ambiguity?

CRI1b — Organizing Information: Students record their four solutions in a personal reference table: starting representation, value of k, equation, and a brief verbal description. This table becomes a study resource.
Assessment: Evaluate whether entries are internally consistent and clearly organized.

Activity — Graph Matching (10 min)
Students receive a set of six cards: three equations and three graphs. They must match each equation to its graph and explain their reasoning in writing. One pair is intentionally mismatched in the card set — students must identify it.

CRI2a — Assessing Validity: The mismatched pair requires students to cross-check the graph against the equation rather than accepting the pairing at face value.
Assessment: Look for students who substitute values from the equation into the graph to verify, rather than relying on visual approximation.

CRI2b — Assessing Quality of Information: After matching, students rank the three equations by "how easy it is to identify k" and justify their ranking.
Assessment: Listen for reasoning about the form of the equation (e.g., "this one already has k isolated, so it's clearest").

CRI3a — Weighing Alternatives: Students are asked: "Would you rather start from a table, a graph, or a verbal description to write an equation? What are the advantages and risks of each?" Small group discussion followed by whole-class share.
Assessment: Note whether students identify specific risks (e.g., "a graph can be hard to read precisely at non-integer points").

Independent Practice (10 min)
Students complete five problems independently, moving between all representations. The final problem asks them to create their own proportional relationship, define k, and represent it in three ways of their choosing.

CRI4a — Applying Sound Reasoning: The creation task requires students to work backward from a chosen k value and verify that all three representations are consistent with each other.
Assessment: Check for internal consistency across representations and ask students to explain their verification process.

Closing — Exit Ticket (5 min)
Two questions: (1) A graph passes through (0,0) and (3,12). Write the equation. (2) Describe one moment today where you caught yourself thinking something incorrect and corrected it.

CRI5a — Reflecting on Thinking: Question 2 specifically prompts students to identify a moment of cognitive correction — a higher-order metacognitive move than simply describing what they learned.
Assessment: Look for specificity and honesty. A student who writes "I thought k was the y-value, not the ratio" demonstrates more genuine reflection than one who writes "I learned about proportions."

Materials: Printed card sets for graph matching, personal reference table template, exit ticket slips.
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

    console.log(`  hallucinations_found : ${result.parsed_review?.hallucinations_found}`);
    console.log(`  types                : ${(result.parsed_review?.types || []).join(", ") || "none"}`);
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