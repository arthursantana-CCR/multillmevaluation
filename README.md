# Multi-LLM Evaluation Pipeline

A configurable pipeline for generating and evaluating educational content using multiple large language models in sequence or in parallel. Designed to reduce hallucinations, improve output reliability, and support longitudinal benchmarking over time.

---

## What This Tool Does

Instead of relying on a single model, this pipeline routes tasks through multiple LLMs in structured architectures. Each model either refines, reviews, or synthesizes outputs from the others. All results are stored as JSON for analysis and comparison over time.

The system is aligned with the Center for Curriculum Redesign (CCR) framework and supports tasks such as lesson plan generation and student writing evaluation.

---

## Two Architectures

### Sequential (Refinement Pipeline)

A generator model produces an initial response. Three reviewer models then evaluate it in sequence, each checking for hallucinations and passing a corrected version to the next.

```
Generator -> Reviewer 1 -> Reviewer 2 -> Final Reviewer -> Final Output
```

If a model fails at any step, the pipeline skips it and passes the last valid output forward.

### Consensus (Synthesis Pipeline)

Three generator models produce independent responses in parallel. An aggregator model then evaluates all three, checks each for hallucinations section by section, and synthesizes a final response drawing the best parts from each candidate.

```
Generator 1 -> Candidate A
Generator 2 -> Candidate B  ->  Aggregator -> Synthesized Final Output
Generator 3 -> Candidate C
```

---

## Repository Structure

```
eval_config.yaml               Main configuration file. This is the primary file you edit.
scripts/
  run_multillm_pipeline.mjs    Pipeline logic: routing, model calls, parsing, saving results.
  render_results.mjs           Renders results/latest.json into a readable Markdown file.
knowledge/
  hallucination/
    rubric.yaml                Hallucination detection rubric used by all reviewers.
  ccr/
    critical_thinking.yaml     CCR competency frameworks (loaded on demand via placeholders).
    collaboration.yaml
    creativity.yaml
    communication.yaml
    courage.yaml
    resilience.yaml
    ethics.yaml
    metacognition.yaml
results/
  latest.json                  Most recent run output.
  latest.md                    Most recent run rendered as Markdown.
  history/                     Immutable JSON history of all runs.
  history_md/                  Immutable Markdown history of all runs.
.github/workflows/
  run_pipeline.yml             GitHub Actions workflow (scheduled or manual).
package.json                   Node.js dependencies.
```

---

## Setup

### Step 1 — Clone the Repository

Clone this repository to your own GitHub account. From the repository page, click Code and then clone it. All workflow automation runs from within GitHub, so no local installation is required unless you want to run the pipeline locally.

### Step 2 — Add API Keys as Repository Secrets

The pipeline calls the Anthropic, Google, and OpenAI APIs. Store your credentials as GitHub Secrets so they are never written into code or committed to Git.

Navigate to:

```
Repository -> Settings -> Secrets and variables -> Actions
```

Click "New repository secret" and add the following:

| Secret Name | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Console -> API Keys |
| `GEMINI_API_KEY` | Google AI Studio -> API Keys |
| `OPENAI_API_KEY` | OpenAI Dashboard -> API Keys |

### Step 3 — Enable GitHub Actions

Scheduled workflows are disabled by default in cloned repositories.

Go to:

```
Repository -> Actions
```

If workflows are disabled, click "Enable workflows."

Run the workflow once manually to verify the setup:

```
Actions -> Run Pipeline -> Run workflow
```

After the first successful run, verify that:
- A new file appears in `results/history/`
- `results/latest.json` has updated
- `results/latest.md` has updated

Once verified, the scheduled runs will activate automatically.

---

## Configuration Reference

All evaluation configuration lives in `eval_config.yaml`. This is the only file most users will need to edit between runs.

### Top-Level Fields

| Field | Values | Description |
|---|---|---|
| `description` | string | Human-readable label for this run. |
| `task_type` | `generation` or `evaluation` | Generation tasks require no cases. Evaluation tasks require cases. |
| `pipeline.architecture` | `sequential` or `consensus` | Which pipeline architecture to use. |
| `hallucination_rubric` | file path | Path to the hallucination rubric YAML. Do not change unless you create a new rubric. |
| `knowledge` | file paths | Registry of CCR framework files. Referenced via placeholders in the task prompt. |
| `task` | string | The instruction sent to generator models. Supports knowledge placeholders. |
| `cases` | list | Required for evaluation tasks. Each case provides an input text to evaluate. |
| `parameters.temperature` | number | Set to 0 for deterministic outputs. |
| `parameters.max_tokens` | number | Maximum tokens per model response. |

### Models

The `pipeline.models` block configures the sequential architecture. The `pipeline.consensus` block configures the consensus architecture. Only the block matching your chosen architecture is used.

```yaml
pipeline:
  architecture: sequential

  models:
    - role: generator
      provider: anthropic
      model: "claude-sonnet-4-6"
    - role: reviewer_1
      provider: google
      model: "gemini-2.5-flash"
    - role: reviewer_2
      provider: openai
      model: "gpt-4.1-mini"
    - role: final_reviewer
      provider: anthropic
      model: "claude-sonnet-4-6"
```

```yaml
pipeline:
  architecture: consensus

  consensus:
    generators:
      - provider: anthropic
        model: "claude-sonnet-4-6"
      - provider: google
        model: "gemini-2.5-flash"
      - provider: openai
        model: "gpt-4.1-mini"
    aggregator:
      provider: anthropic
      model: "claude-sonnet-4-6"
```

Supported providers: `anthropic`, `google`, `openai`.

---

## Task Types

### Generation

No cases required. The pipeline runs the task prompt once and produces a single output.

```yaml
task_type: generation

task: |
  Generate a 55-minute lesson plan for 7th grade students on the topic of climate change.
  The lesson should be suitable for a traditional classroom setting.
```

### Evaluation

Requires one or more cases. Each case provides an input text. The pipeline evaluates each case independently and produces a separate output per case.

```yaml
task_type: evaluation

task: |
  Evaluate the following student text using the CCR Critical Thinking framework.
  For each subcompetency, assign a level and cite evidence from the student text.

cases:
  - id: "student_001"
    input: |
      Climate change is a serious problem. Scientists say the earth is getting warmer
      because of carbon emissions. We should use renewable energy to fix this.

  - id: "student_002"
    input: |
      I think homework is bad because it takes too long and I don't have time for sports.
```

---

## Knowledge Layer

The knowledge layer stores CCR competency frameworks and the hallucination rubric as separate YAML files. This keeps prompts clean and makes frameworks easy to update without touching pipeline logic.

### Hallucination Rubric

Located at `knowledge/hallucination/rubric.yaml`. This file is loaded automatically at startup and injected into every reviewer and aggregator prompt. You do not need to reference it in your task prompt.

To update the rubric, edit that file directly. The change applies to all future runs.

### CCR Frameworks

Located in `knowledge/ccr/`. Each file defines a competency with subcompetencies and scoring levels.

These are loaded on demand. To include a framework in your task prompt, use a placeholder:

```yaml
task: |
  Evaluate the following student text using the CCR Critical Thinking framework below.

  {{ccr.critical_thinking}}

  For each subcompetency, assign a level and cite evidence from the student text.
```

Available placeholders:

| Placeholder | File |
|---|---|
| `{{ccr.critical_thinking}}` | knowledge/ccr/critical_thinking.yaml |
| `{{ccr.collaboration}}` | knowledge/ccr/collaboration.yaml |
| `{{ccr.creativity}}` | knowledge/ccr/creativity.yaml |
| `{{ccr.communication}}` | knowledge/ccr/communication.yaml |
| `{{ccr.courage}}` | knowledge/ccr/courage.yaml |
| `{{ccr.resilience}}` | knowledge/ccr/resilience.yaml |
| `{{ccr.ethics}}` | knowledge/ccr/ethics.yaml |
| `{{ccr.metacognition}}` | knowledge/ccr/metacognition.yaml |

You may include multiple placeholders in the same task prompt. Only the frameworks you reference are loaded, keeping token usage minimal.

To add a new framework, create a YAML file in `knowledge/ccr/` following the same structure as existing files, then add an entry to the `knowledge.ccr` block in `eval_config.yaml`.

---

## Output Format

Each run produces a JSON result file and a rendered Markdown file.

### JSON Structure

```
run_id
run_time_utc
architecture
model_sequence
cases[]
  case_id
  prompt
  outputs
    [sequential]
      generator_output.raw_text
      reviewer_1_output.parsed_review
      reviewer_2_output.parsed_review
      final_reviewer_output.parsed_review
      final_output
    [consensus]
      candidate_1.raw_text
      candidate_2.raw_text
      candidate_3.raw_text
      final_output
        sources_used
        hallucinations_found
        types
        justification
        corrected_answer
```

### Hallucination Fields

Every reviewer and aggregator output includes:

| Field | Description |
|---|---|
| `hallucinations_found` | Boolean. Whether any hallucinations were detected. |
| `types` | Array. Classification codes: A1, A2, B1, B2. |
| `justification` | String. Brief explanation of the hallucination assessment. |
| `corrected_answer` | String. The output with hallucinated content removed or corrected. |

The consensus aggregator also includes:

| Field | Description |
|---|---|
| `sources_used` | Array. Which candidates contributed to the final synthesized output, e.g. ["A", "B"]. |

---

## Example: Full Generation Config

```yaml
description: "Lesson plan generation with CCR Critical Thinking"

task_type: generation

pipeline:
  architecture: consensus

  consensus:
    generators:
      - provider: anthropic
        model: "claude-sonnet-4-6"
      - provider: google
        model: "gemini-2.5-flash"
      - provider: openai
        model: "gpt-4.1-mini"
    aggregator:
      provider: anthropic
      model: "claude-sonnet-4-6"

parameters:
  temperature: 0
  top_p: 1.0
  max_tokens: 7000

hallucination_rubric: knowledge/hallucination/rubric.yaml

knowledge:
  ccr:
    critical_thinking: knowledge/ccr/critical_thinking.yaml

task: |
  You are an expert educator. Use the CCR Critical Thinking framework below to design
  a 55-minute lesson plan for 7th grade students.

  {{ccr.critical_thinking}}

  For each subcompetency, include a specific activity and how you would assess it.
  The lesson topic is: "Should schools have homework?"
```

## Example: Full Evaluation Config

```yaml
description: "Student writing evaluation with CCR Critical Thinking"

task_type: evaluation

pipeline:
  architecture: sequential

  models:
    - role: generator
      provider: anthropic
      model: "claude-sonnet-4-6"
    - role: reviewer_1
      provider: google
      model: "gemini-2.5-flash"
    - role: reviewer_2
      provider: openai
      model: "gpt-4.1-mini"
    - role: final_reviewer
      provider: anthropic
      model: "claude-sonnet-4-6"

parameters:
  temperature: 0
  top_p: 1.0
  max_tokens: 4000

hallucination_rubric: knowledge/hallucination/rubric.yaml

knowledge:
  ccr:
    critical_thinking: knowledge/ccr/critical_thinking.yaml

task: |
  Evaluate the following student text using the CCR Critical Thinking framework below.

  {{ccr.critical_thinking}}

  For each subcompetency, assign a level (-1 to 2) and cite evidence from the student text.

cases:
  - id: "student_001"
    input: |
      Climate change is caused by humans burning fossil fuels. Scientists have proven
      this many times. We should switch to solar and wind energy because they are cleaner.
      Some people disagree but they are wrong because the evidence is clear.
```

---

## Scheduling

The workflow schedule is defined in `.github/workflows/run_pipeline.yml`.

To change the schedule, modify the `cron` expression in that file. Times are in UTC.

Example: run daily at 10:00 UTC:

```yaml
on:
  schedule:
    - cron: "0 10 * * *"
  workflow_dispatch:
```

`workflow_dispatch` enables manual triggering from the Actions tab at any time.

---

## Running Locally

To run the pipeline locally:

```bash
npm install
ANTHROPIC_API_KEY=your_key GEMINI_API_KEY=your_key OPENAI_API_KEY=your_key node scripts/run_multillm_pipeline.mjs
```

Results are written to `results/latest.json` and `results/latest.md`.
