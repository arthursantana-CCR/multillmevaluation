export function buildRuntimeConfig({ prompt, architecture }) {
  return {
    description: "UI runtime execution",

    task_type: "generation",

    pipeline: {
      architecture,

      models: [
        {
          role: "generator",
          provider: "anthropic",
          model: "claude-sonnet-4-6"
        },
        {
          role: "reviewer_1",
          provider: "google",
          model: "gemini-2.5-flash"
        },
        {
          role: "reviewer_2",
          provider: "openai",
          model: "gpt-4.1-mini"
        },
        {
          role: "final_reviewer",
          provider: "anthropic",
          model: "claude-sonnet-4-6"
        }
      ],

      consensus: {
        generators: [
          { provider: "anthropic", model: "claude-sonnet-4-6" },
          { provider: "google", model: "gemini-2.5-flash" },
          { provider: "openai", model: "gpt-4.1-mini" }
        ],
        aggregator: {
          provider: "anthropic",
          model: "claude-sonnet-4-6"
        }
      }
    },

    parameters: {
      temperature: 0,
      top_p: 1.0,
      max_tokens: 3500
    },

    system_instruction: `
You are part of a multi-step AI generation pipeline.

Your output MUST follow this EXACT JSON structure:

{
  "hallucinations_found": boolean,
  "types": string[],
  "justification": string,
  "corrected_answer": string
}

RULES:
- Output ONLY valid JSON
- Do NOT include markdown, headings, or explanations
- Do NOT include "FINAL ANSWER" or any extra text
- "corrected_answer" MUST always be present
- If no hallucinations are found, return the original answer unchanged in "corrected_answer"
`,

    hallucination_rubric: `
A hallucination is defined as any instance where the model presents information as factual, sourced, or real without verifiable grounding.

The following MUST be classified as hallucinations:
1. Fabricated references
2. Unsupported claims presented as factual
3. Specific numerical claims without grounding

The following are NOT hallucinations:
- Clearly fictional examples
- Hypothetical scenarios
- General pedagogical explanations
`,

    // 🔥 CRITICAL: UI injects prompt here
    task: prompt,

    cases: [
      {
        id: "ui_case",
        input: "N/A"
      }
    ],

    output_format: {
      type: "json_only"
    }
  };
}
