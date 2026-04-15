async function runPipeline() {
  const prompt = document.getElementById("prompt").value;
  const architecture = document.getElementById("architecture").value;

  const output = document.getElementById("output");

  output.textContent = "Running...";

  try {
    const res = await fetch("/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ prompt, architecture })
    });

    const data = await res.json();

    if (data.error) {
      output.textContent = data.error;
    } else {
      // 🔥 Proper formatting (preserves line breaks)
      output.textContent = data.final_output;
    }

  } catch (err) {
    output.textContent = "Error running pipeline.";
  }
}
