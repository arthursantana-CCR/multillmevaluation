import os
import pandas as pd
from datetime import datetime
from litellm import completion

# Models to test
MODELS = ["gpt-4o", "gemini/gemini-1.5-flash"]
PROMPT = "what is the color of the sky?"

def run_benchmark():
    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for model in MODELS:
        try:
            print(f"Testing {model}...")
            response = completion(
                model=model, 
                messages=[{"role": "user", "content": PROMPT}]
            )
            answer = response.choices[0].message.content
            results.append({
                "date": timestamp,
                "model": model,
                "response": answer.replace("\n", " ") # Keep it on one line for CSV
            })
        except Exception as e:
            print(f"Error testing {model}: {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    file_exists = os.path.isfile("results.csv")
    df.to_csv("results.csv", mode='a', index=False, header=not file_exists)

if __name__ == "__main__":
    run_benchmark()