# LLM Test Automation (test)
### Promptfoo + GitHub Actions + Airtable

This repository runs automated daily evaluations of multiple LLMs on the **same prompt suite**, stores raw evaluation results in Git, and logs a summary row per run in Airtable.

The goal is to track model behavior longitudinally — allowing teams to observe how different models perform on the same tasks over time.

The system was designed to support both simple prompt tests and complex evaluation tasks, such as automated scoring of student competencies.

---

## Repository Purpose

This repository allows you to:

- Run the same prompts daily against multiple AI models
- Track changes in model behavior over time
- Record outputs, pass/fail status, and token usage
- Maintain full historical JSON records
- Log summarized results to Airtable for analysis

This makes it possible to monitor: model drift, response structure stability, cost trends, latency changes, and output verbosity.

---

## Getting Started

Git is a version control system that tracks changes to files over time. GitHub hosts Git repositories online, allowing collaboration and automation.

To use this project, first clone the repository. From the repository page, click **Code** → **Clone repository**. You may clone to your local machine or to a new GitHub repository under your account.

After cloning, you must configure API credentials and enable GitHub Actions. Scheduled workflows will not run until this setup is completed.

---

## Step 1 — Add Required API Keys as Repository Secrets

The pipeline calls external APIs (OpenAI, Anthropic, Airtable). These credentials must be stored securely as GitHub Secrets — encrypted values injected at runtime, never written into code or committed to Git.

Navigate to:
```
Repository → Settings → Secrets and variables → Actions
```

Click **New repository secret** and add the following:

| Secret Name | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AIRTABLE_PAT` | Airtable Personal Access Token |
| `AIRTABLE_BASE_ID` | Airtable Base ID |
| `AIRTABLE_TABLE_NAME` | Airtable table name (e.g. `DailyRuns`) |

### Where to obtain these values

**OpenAI** — OpenAI Dashboard → API Keys

**Anthropic** — Anthropic Console → API Keys

**Airtable PAT:**
1. Go to the Airtable Developer Hub.
2. Create a Personal Access Token.
3. Required scopes: `data.records:write` (also recommended: `data.records:read`).
4. Grant access to your Airtable base.
5. Copy the PAT — it is shown only once.

**Airtable Base ID** — found in the Airtable URL:
```
https://airtable.com/appXXXXXXXXXXXXXX/...
```

---

## Step 2 — Configure Airtable Fields

Your Airtable table must include these fields with **exactly** these names:

| Field | Type |
|---|---|
| `RunID` | Text |
| `RunTimeUTC` | Date + Time |
| `Prompt` | Long text |
| `OpenAI_Model` | Text |
| `OpenAI_Output` | Long text |
| `OpenAI_Passed` | Checkbox |
| `Claude_Model` | Text |
| `Claude_Output` | Long text |
| `Claude_Passed` | Checkbox |
| `OpenAI_Input_Tokens` | Number |
| `OpenAI_Output_Tokens` | Number |
| `OpenAI_Total_Tokens` | Number |
| `Claude_Input_Tokens` | Number |
| `Claude_Output_Tokens` | Number |
| `Claude_Total_Tokens` | Number |
| `GitHub_Run_URL` | URL |

The token fields track API usage per run and help estimate costs for larger experiments.

---

## Step 3 — Confirm Required Files

The repository should contain:

| File | Purpose |
|---|---|
| `eval_config.yaml` | User configuration for prompts and models |
| `.github/workflows/daily-llm-eval.yml` | Scheduled workflow |
| `scripts/build_promptfoo_config.mjs` | Generates Promptfoo config automatically |
| `scripts/post_to_airtable.mjs` | Posts results to Airtable |
| `package.json` | Node dependencies |

Your `.gitignore` should include:
```
node_modules/
promptfooconfig.yaml
```

`promptfooconfig.yaml` is generated automatically and should never be committed.

---

## Step 4 — Configure Your Evaluation

All evaluation configuration lives in `eval_config.yaml`. Most users will only ever edit this file.

### Example 1 — Simple Prompt Test
```yaml
description: "Daily prompt suite"

openai_model: "openai:chat:gpt-4.1-mini"
anthropic_model: "anthropic:messages:claude-sonnet-4-20250514"

temperature: 0
top_p: 1.0
max_tokens: 256

cases:
  - id: "sky_blue"
    prompt: "What is the color of the sky?"
    assert_contains: "blue"
```

### Example 2 — Multiple Prompt Tests
```yaml
cases:
  - id: "sky_blue"
    prompt: "What is the color of the sky?"
    assert_contains: "blue"

  - id: "grass_green"
    prompt: "What color is grass?"
    assert_contains: "green"

  - id: "banana_yellow"
    prompt: "What color is a ripe banana?"
    assert_contains: "yellow"
```

Each case runs against every configured model.

### Example 3 — Complex Evaluation Task (Education)

The system also supports structured evaluation tasks such as automated rubric scoring. You can provide a rubric, a task instruction, and a student text:

| Field | Purpose |
|---|---|
| `rubric` | Evaluation framework |
| `task` | Instructions for the model |
| `student_text` | Content to evaluate |
```yaml
rubric: |
  CCR Critical Thinking rubric...

task: |
  Evaluate the student text using the rubric.

cases:
  - id: "student_001"
    student_text: |
      Student essay text...
```

The pipeline automatically constructs the full prompt.

### Assertions

Assertions verify whether the model output meets expectations. A case can require multiple assertions:
```yaml
assert_contains:
  - "\"CRI1\""
  - "\"CRI2\""
  - "\"CRI3\""
  - "\"CRI4\""
  - "\"CRI5\""
```

This ensures required output fields exist. Assertions check structure, not correctness.

### Generation Parameters

| Parameter | Purpose |
|---|---|
| `temperature` | Randomness in output generation |
| `top_p` | Nucleus sampling threshold |
| `max_tokens` | Maximum response length |

We set `temperature: 0` and `top_p: 1.0` to minimize randomness and ensure consistent results across runs.

### System Instruction

You may optionally define a `system_instruction` to standardize behavior across models and prevent provider defaults from affecting results:
```yaml
system_instruction: |
  Follow the instructions exactly.
  Return only the requested format.
```

---

## Step 5 — Enable GitHub Actions

GitHub Actions is GitHub's built-in automation system. It runs workflows automatically on a schedule or when triggered manually. In this repository, GitHub Actions is responsible for running the daily LLM evaluations, posting results to Airtable, and committing result files back to Git — all without manual intervention once set up.

Scheduled workflows are disabled by default in newly created or copied repositories. You must enable them manually.

Go to:
```
Repository → Actions
```

If workflows are disabled, click **Enable workflows**.

### Run the Workflow Once Manually

Scheduled workflows may not activate until they have been run at least once manually. Go to:
```
Actions → Daily LLM Eval
```

Click **Run workflow**, then verify:
- A new JSON file appears in `results/history/`
- `results/latest.json` has updated
- A new row appears in Airtable

After the first successful run, the daily schedule activates automatically.

---

## Where Results Are Stored

**In Git**, each run produces:
- `results/history/<timestamp>.json` — immutable run history
- `results/latest.json` — most recent run
- `results/latest_timestamp.txt`

These files are committed automatically by the workflow.

**In Airtable**, each run creates one row containing: timestamp, model IDs, outputs, pass/fail status, token usage, and the GitHub run link.

---

## How the Pipeline Works

1. GitHub Actions triggers the workflow (daily or manual).
2. `eval_config.yaml` is read.
3. `build_promptfoo_config.mjs` generates `promptfooconfig.yaml`.
4. Promptfoo executes the evaluation.
5. Raw JSON results are written to `results/`.
6. `post_to_airtable.mjs` posts a summary row to Airtable.
7. Results are committed back to Git.

---

## Scheduling

The workflow schedule is defined in `.github/workflows/daily-llm-eval.yml`.

Default: **10:00 GMT-3**

To change the schedule, modify the `cron` expression in that file.

---

## What Users Usually Change

Most users only modify `eval_config.yaml`. Common edits include: prompts or student texts, expected assertions, models under test, and generation parameters.

Everything else in the repository is infrastructure.
