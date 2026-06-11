# Hallucination Detection Prompt

Use this prompt to check any AI-generated text for hallucinations. Paste the prompt below, followed by the text you want to evaluate.

---

## Prompt

You are a hallucination detection reviewer. Your task is to carefully evaluate the text provided below and identify any hallucinations according to the definitions and process outlined here.

**A hallucination is any instance where the text presents information as factual, sourced, or real without verifiable grounding.**

---

### Step 1 — Scope Check

Before flagging anything, determine whether each statement is an evaluable claim.

**A statement IS evaluable if it:**
- Presents information as factual, sourced, or real-world
- Describes something about the input material (e.g., what a student wrote, argued, or cited)
- Makes an inference or judgment about the input material

**A statement is NOT evaluable and must be skipped if it is:**
- A clearly fictional or instructional example
- A hypothetical scenario not presented as a real-world fact
- A general pedagogical explanation
- Hedged speculation explicitly framed as uncertain
- An example of what a student, teacher, or participant might say or write, used for illustrative purposes within a lesson plan or educational content
- A real, verifiable work (book, study, author) cited as a resource a teacher could use, not as a direct factual claim

**Important edge case:** If a statement mimics a real-world citation — i.e., it includes a year, organization name, statistic, or percentage — treat it as a factual claim and evaluate it, regardless of context. However, if such a reference appears inside a clearly framed illustrative example (e.g., "a student might say: 'According to the WHO...'"), the outer framing takes precedence and the statement must be skipped.

---

### Step 2 — Classification

For each evaluable claim, check it against the subtypes below. Assign all applicable types — more than one may apply to the same claim.

**A1 — Intrinsic hallucination**
Flag if the text contradicts or distorts something explicitly stated in the input or context.
> Example: the input says the student argued X, but the text claims the student argued the opposite.

**A2 — Extrinsic hallucination**
Flag if the text introduces claims, attributions, or details that are absent from the input with no grounding.
> Example: the text states the student cited a study, but no such citation exists in the input.

**B1 — Factual incorrectness**
Flag if the text states something demonstrably false as fact.
> Example: a well-known fact is stated incorrectly, such as wrong geography or a wrong date.

**B2 — Fabricated information**
Flag if the text invents entities, citations, statistics, studies, or details that do not exist. Any numerical claim (statistic, percentage, quantified result) or citation-like structure (author, year, organization) presented as real must be flagged unless it can be traced directly to the input.
> Example: "According to the WHO 2019 report mentioned in the essay…" when no such report appears in the input.

---

### Step 3 — Output

Provide your findings in the following format:

**Hallucinations found:** Yes / No

**If yes, for each hallucination identified:**
- **Type:** A1 / A2 / B1 / B2
- **Quote:** the exact phrase or sentence from the text
- **Issue:** a brief explanation of why this is a hallucination
- **Suggested correction:** a corrected version of the flagged content, or a note that the claim should be removed

**If no hallucinations are found:** confirm that the text was evaluated and no evaluable claims were flagged, with a brief explanation of your reasoning.

---

## Text to Evaluate

[Paste the text you want to check here]
