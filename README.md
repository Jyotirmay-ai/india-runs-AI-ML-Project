# AI Candidate Ranking System

**The Data & AI Challenge** - Intelligent Candidate Discovery & Ranking

## Overview
A two-stage Retrieval-Augmented Ranking (RAR) system that goes beyond keyword matching to evaluate genuine candidate fit for a Senior AI Engineer role at Redrob AI.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐       ┌──────────────────────┐
│  100,000        │     │  71,813          │       │  Top 100             │
│  Candidates     │────▶│  After Behavioral│────▶ │  Semantic Search     │
│  (JSONL)        │     │  Pre-filtering   │       │  (Hybrid TF-IDF +    │
└─────────────────┘     └──────────────────┘       │  SBERT)              │ 
                                                   └──────────┬───────────┘
                                                              │
                                                   ┌──────────▼─────────────┐
                                                   │  Top 100               │
                                                   │  LLM Recruiter         │
                                                   │  Scoring (Groq/        │
                                                   │  Llama3-3.1-8b-instant)│
                                                   └──────────┬─────────────┘
                                                              │
                                                   ┌──────────▼────────────┐
                                                   │  Final Ranked         │
                                                   │  Output + Rationale   │
                                                   └───────────────────────┘ 
```

## Key Innovations

### 1. Behavioral Pre-filtering (Stage 0)
- Filters on `redrob_signals`: `last_active_date` (within 6 months) + `recruiter_response_rate` (>5%)
- Removes 28% of candidates who are "unhirable" per JD criteria before any AI compute

### 2. Career-History Semantic Search (Stage 1)
- **Avoids the keyword trap**: Embeds full `career_history` narratives, not skill arrays
- **Hybrid retrieval**: TF-IDF (broad recall) → SBERT `all-MiniLM-L6-v2` (precision)
- Processes 71k→5k→100 candidates efficiently with FAISS

### 3. LLM "Recruiter" Scoring (Stage 2)
- Prompts Llama3-70b (via Groq) with full JD context and candidate narrative
- Scores 4 dimensions: Hard Skills, Experience Relevance, Behavioral Fit, Shipper Mindset
- Outputs score + reasoning + red flags for explainability

## Tech Stack
- **Language**: Python 3.10+
- **Data**: Pandas, NumPy
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS (CPU)
- **LLM**: Groq API (llama-3.1-8b-instant) - free tier
- **Orchestration**: Custom pipeline

## Portable Setup & Quick Start

To use this project on a new machine, ensure the following folder structure for the dataset placement:

<img src="dataset placement style demo.png" alt="Dataset Placement Demo" width="600"/>

```
india-runs-ai-project/
├── dataset/
│   └── dataset_provided/          # Rename your dataset folder to this
│       └── candidates.jsonl       # Must contain the candidates data
├── project/                       # Clone the project folder here
│   ├── src/
│   ├── testing/                    # Benchmark and test scripts
│   ├── logs/                       # Execution logs (created automatically)
│   ├── jd_text.txt
│   ├── main.py
│   └── .env                       # Create from .env.example
└── Result/                        # Created automatically
```

### Setup Steps

1. **Create and Activate a Virtual Environment**
   Navigate to the `project` folder and set up a virtual environment:
   ```bash
   cd project
   python -m venv .venv
   
   # Activate (Windows)
   .venv\Scripts\activate
   # Activate (Mac/Linux)
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r project/requirements.txt
   ```

3. **Configure API Key**
    - Copy the template: `cp .env.example .env` (or rename it manually)
    - Open `.env` and add your key: `GROQ_API_KEY=your_key_here`

4. **Run full pipeline**
    ```bash
    # Run from the project directory
    cd project
    python main.py
    ```
    The system will prompt you to enter how many top candidates you want in the final spreadsheet (default: 60, max: 100). Just press Enter for default or type a number.
    *The system automatically calculates and prints the execution time for each stage and the total processing time. All operations are logged to timestamped files in the `logs/` directory for debugging and audit purposes.*

## Autonomous Output Structure
Each pipeline run creates a **timestamped folder** in `Result/`:
```
Result/2026-06-23_14-30-45/
├── 1_filtered_candidates.jsonl      # Stage 0: After behavioral pre-filter (71k candidates)
├── 2_shortlisted_candidates.jsonl   # Stage 1: Top 100 from semantic search
├── 3_ranked_candidates.jsonl        # Stage 2: Full LLM evaluation details (100 candidates)
├── 5_ranking_progress.jsonl         # Stage 2: Checkpoint for resume capability
├── 6_ranked_candidates.csv          # Top N candidates (CSV spreadsheet)
└── 6_ranked_candidates.xlsx         # Top N candidates (Excel spreadsheet)
```

## Outputs (per run folder)
| File                                    |               Description                              |
|-----------------------------------------|--------------------------------------------------------|
| `1_filtered_candidates.jsonl`           | After behavioral pre-filter (~71k candidates)          |
| `2_shortlisted_candidates.jsonl`        | Top 100 from semantic search                           |
| `3_ranked_candidates.jsonl`             | Full LLM evaluation details (100 candidates)           |
| `5_ranking_progress.jsonl`              | Incremental checkpoint for resume capability           |
| `6_ranked_candidates.csv`               | Top N ranked candidates (spreadsheet, configurable)    |
| `6_ranked_candidates.xlsx`              | Top N ranked candidates (Excel, configurable)          |

## Evaluation Criteria (from JD)
1. **Production Embeddings/Retrieval** - Built & deployed to real users
2. **Vector DB / Hybrid Search** - Pinecone, FAISS, Elasticsearch, etc.
3. **Strong Python** - Production code quality
4. **Ranking Evaluation** - NDCG, MRR, MAP, A/B testing
5. **Pre-LLM ML Depth** - Retrieval/ranking before it was fashionable
6. **Product Company** - Not pure consulting/services
7. **Shipper Mindset** - Ship v2 ranker in weeks, not research-only

## Results based on the dataset provided
- **Top Candidate**: Aarohi Bose (CAND_0093193) - Score 92
  - Built ranking/retrieval at Niramai & Netflix
  - Handles embedding drift, index refresh in production
- **Runner-up**: Aryan Goyal (CAND_0005538) - Score 92
  - Embedding-based retrieval at Adobe & Locobuzz
  - "Shipper" attitude: boring-but-essential infra work

## Changing Datasets & Job Descriptions
When using this system for a completely new hiring round, you must do two things:
1. **Update the Job Description:** Open `project/jd_text.txt` and replace its contents with the new job description. If you forget this, the AI will rank your new candidates based on the old role!
2. **Clear the Cache:** The system caches vector embeddings in the `project/cache/` directory to save compute time. If you swap out `candidates.jsonl` for a completely *new* dataset, you **must delete the `project/cache/` folder** before running `main.py`. Otherwise, the system will incorrectly load the old candidates' embeddings!

## Reproducibility
- All intermediate artifacts cached (`project/cache/`) - shared across runs
- Incremental checkpointing in Stage 2 (`5_ranking_progress.jsonl`)
- Deterministic TF-IDF + fixed model weights

## License
MIT
