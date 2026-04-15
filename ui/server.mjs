import express from "express";
import path from "path";
import { fileURLToPath } from "url";

import { runPipelineWithConfig } from "../scripts/run_multillm_pipeline.mjs";
import { buildRuntimeConfig } from "./build_runtime_config.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.post("/run", async (req, res) => {
  try {
    const { prompt, architecture } = req.body;

    const config = buildRuntimeConfig({
      prompt,
      architecture
    });

    const result = await runPipelineWithConfig(config);

    // 🔥 CORRECT parsing (handles both consensus + sequential)
    const raw = result.cases?.[0]?.outputs?.final_output;

    let finalText = "";

    try {
      const parsed = JSON.parse(raw);

      if (parsed.corrected_answer) {
        finalText = parsed.corrected_answer;
      } else {
        finalText = raw;
      }

    } catch (e) {
      // sequential case (already plain text)
      finalText = raw;
    }

    res.json({
      final_output: finalText
    });

  } catch (err) {
    console.error(err);

    res.status(500).json({
      error: "Pipeline execution failed"
    });
  }
});

app.listen(3000, () => {
  console.log("UI running on http://localhost:3000");
});
