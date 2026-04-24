# LLM-as-a-Judge for Paraphrase Evaluation

This project uses an LLM to evaluate paraphrases.

For each sentence pair, it scores:
- Semantic Equivalence
- Fluency
- Diversity

Each score is from 1-5, along with a short explanation.

---

## Project Structure

project/
├── main.py  
├── analysis.py  
├── *_predictions.csv  
├── *_llm_results.csv  
├── requirements.txt  
└── .env  

---

## Input

CSV with columns:

source,prediction  

---

## Output

CSV with columns:

source,prediction,llm_semantic_equivalence,llm_fluency,llm_diversity,llm_justification  

---

## How to Run

1. Create environment  
python -m venv judge_env  

2. Activate (Windows PowerShell)  
.\judge_env\Scripts\Activate  

3. Install dependencies  
pip install -r requirements.txt  

4. Add API key in `.env`  
OPENAI_API_KEY=your_api_key_here  

5. Run  
python main.py  

6. Analyze  
python analysis.py  

---

## Config

In `main.py`:

run_csv_pipeline(
    input_csv="your_file.csv",
    output_csv="your_output.csv",
    source_col="source",
    candidate_col="prediction",
    limit=50
)

limit controls how many rows are processed.

---

## Analysis

In `analysis.py`, set the filenames in the `files` dictionary:

files = {
    "Model Name": "your_results.csv",
}

Then run:

python analysis.py