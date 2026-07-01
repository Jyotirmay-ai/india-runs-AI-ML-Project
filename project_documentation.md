# Project Documentation: AI Candidate Ranking System

This document provides a comprehensive explanation of how the AI Candidate Ranking System works and a detailed breakdown of the project's components.

---

## 1. How the System Works (The Big Picture)

Imagine you have 100,000 resumes, but only a few minutes to find the perfect "Senior AI Engineer." Reading every resume is impossible, and searching for simple keywords (like "Python" or "AI") isn't enough because thousands of people have those words on their profiles.

Our system acts like a **highly efficient digital recruiter**. Instead of looking at everything at once, it uses a "funnel" approach—starting with a massive group and narrowing it down in three distinct stages.

### The "Funnel" Process

#### Stage 0: The Behavioral Filter (The "Are they interested?" check)
Before looking at skills, the system checks if the candidate is actually active. If someone hasn't been active on the platform for 6 months or rarely responds to recruiters, they are likely not looking for a job.
- **Action:** It removes candidates based on behavioral signals.
- **Result:** We go from **100,000 → ~72,000 candidates**.

#### Stage 1: Hybrid Semantic Search (The "Do they have the right experience?" check)
Now we need to find the people whose *actual career history* matches the Job Description (JD). We don't just look for keywords; we look for **meaning**.
- **Narrative Building:** The system turns a candidate's fragmented work history into a "career story" (e.g., *"At Company X as an AI Engineer: Built a search system..."*).
- **Two-Step Search:**
    1. **Fast Scan (TF-IDF):** Quickly identifies the top 5,000 candidates who use similar language to the JD.
    2. **Deep Scan (SBERT):** Uses a "Deep Learning" model to understand the context. It converts the JD and the candidate stories into mathematical vectors (embeddings) and finds the 100 candidates whose "story" is mathematically closest to the JD's requirements.
- **Result:** We go from **~72,000 → 100 candidates**.

#### Stage 2: LLM Recruiter Scoring (The "Are they truly an expert?" check)
Now that we have the top 100, we can afford to use a very "smart" but "slow" AI (Groq's Llama-3.1). The LLM acts as an expert technical recruiter.
- **Deep Evaluation:** The LLM reads the full JD and the candidate's full profile. It doesn't just give a score; it evaluates four specific dimensions:
    1. **Hard Skills:** Do they know the specific tech?
    2. **Experience Relevance:** Have they done this in a real production environment?
    3. **Behavioral Fit:** Do they match the company culture?
    4. **Shipper Mindset:** Do they build and ship products quickly?
- **Reasoning:** The AI explains *why* it gave that score, citing specific evidence from the candidate's history.
- **Result:** We get a **ranked list of the best candidates** with detailed justifications.

---

## 2. File-by-File Breakdown

### Core Orchestration
- **`main.py`**: The "Brain" of the project. It imports all the stages, starts the process in the correct order (Stage 0 → 1 → 2), manages the folder for the current run, and prints the final results. It also calculates the exact time each stage takes to complete and logs all operations.
- **`src/config.py`**: The "Map" of the project. Instead of writing paths like `C:\Users\...` everywhere, all files ask this module where to find the dataset, the JD, or the cache. This makes the project portable so it works on any computer.
- **`src/run_manager.py`**: The "Organizer." Its only job is to create a new folder in the `Result/` directory with a timestamp (e.g., `2026-06-24_12-00-00`) so that every time you run the pipeline, your old results aren't overwritten.

### The Processing Pipeline
- **`src/prefilter.py`**: Implements **Stage 0**. It reads the massive `candidates.jsonl` file and applies simple rules (active date and response rate) to filter out candidates who are unlikely to respond.
- **`src/semantic_search.py`**: Implements **Stage 1**. This is the most mathematically complex file. It handles the TF-IDF broad search, uses the Sentence-Transformer model to create "embeddings" (numerical representations of text), and uses FAISS (a fast similarity search library) to find the top 100 matches. It also manages a `cache/` folder so it doesn't have to re-calculate embeddings every time you run it. 

*(Note: If you use a brand new dataset, you must manually delete the `project/cache/` folder before running, or it will load the wrong mathematical representations!)*

- **`src/llm_ranker.py`**: Implements **Stage 2**. It connects to the Groq API and sends a carefully crafted "prompt" (instructions) to the Llama model. It tells the AI exactly how to score the candidates and ensures the output is in a clean JSON format that the computer can save.

### Support & Utilities
- **`src/logger.py`**: The "Recorder." It sets up a comprehensive logging system that creates timestamped log files in the `logs/` directory for every run. It captures all system activities, errors, warnings, and informational messages, making debugging and auditing straightforward.
- **`src/exporter.py`**: The "Exporter." Merges the shortlisted candidate profiles with their LLM evaluation scores, then writes a flat spreadsheet (CSV + Excel) containing the top N candidates. The number of candidates is asked interactively at runtime (default: 60, max: 100).
- **`testing/benchmark_pipeline.py`**: A specialized tool used to measure the performance of the system. It runs the entire pipeline and provides a detailed timing report for each stage, which is useful for optimizing the system.

### Support Files
- **`.env`**: A private file that stores your `GROQ_API_KEY`. It keeps your secret keys out of the main code.
- **`.env.example`**: A template for other users. Since `.env` is private, this tells others which keys they need to provide.
- **`requirements.txt`**: A list of all the Python libraries (like `pandas`, `faiss`, `sentence-transformers`) needed to make the system work.
- **`jd_text.txt`**: The Job Description. This is the "gold standard" the system uses to compare all candidates against. *(Note: If you are using this system to hire for a different role, you must replace the text in this file with your new job description!)*
- **`README.md`**: The "User Manual" for a quick start.

## 3. The "Save Game" File: Resume Capability

The file `4_ranking_progress.jsonl` acts like a **"save game"** for the LLM scoring stage.

**Where it lives:** The progress checkpoint is stored at `project/cache/ranking_progress.jsonl` — a **shared location** that persists across runs. Each run folder also gets a copy (`4_ranking_progress.jsonl`) for record-keeping.

**Why it exists:** Stage 2 calls the Groq API to score 100 candidates one-by-one. This takes several minutes. If something crashes halfway (network error, API rate limit, power outage), you don't want to re-score the candidates already done.

**How it works:**
- After each candidate is scored, one line is appended to the shared progress file
- Next run: the system reads this file, sees which candidates are already done, and **skips them**
- It resumes from where it left off automatically — even if you run `main.py` fresh (which creates a new timestamped folder)

**Example:**
> Run 1: Scores 47 candidates → crash!  
> Run 2: Reads shared progress file, sees 47 done → only scores remaining 53

**Why you need it:** It's insurance. If the Groq API has a hiccup (common on free tier), you just re-run `python main.py` and it picks up automatically. No manual intervention needed.

**Important — When to clear the progress:**
If you change the dataset or job description, the shortlisted candidates may differ. The old progress would be invalid. Clear it by running:
```python
from src.llm_ranker import clear_progress_checkpoint
clear_progress_checkpoint()
```
Or simply delete `project/cache/ranking_progress.jsonl` before the next run.

## 4. Logging & Debugging

The system includes a robust logging mechanism to help with troubleshooting and auditing:
- **Automatic Timestamped Logs:** Each pipeline run creates a new log file in `project/logs/` named like `run_2026-06-24_12-00-00.log`.
- **Multiple Log Levels:** Captures INFO (general operations), WARNING (potential issues), ERROR (failures), and CRITICAL (system-breaking problems).
- **Exception Tracking:** When errors occur, the full stack trace is logged to help pinpoint exactly where and why something failed.
- **Console + File Output:** Logs appear both in the terminal (for immediate feedback) and in the log file (for permanent record).
- **Integration:** All core components (`main.py`, stage scripts) use the logger, ensuring comprehensive coverage.

## Summary Flow
`main.py` → `config.py` (Paths) → `run_manager.py` (Folder) → `prefilter.py` → `semantic_search.py` → `llm_ranker.py` → `exporter.py` (CSV/XLSX) → **Ranked Result!** (with Execution Time Report, Comprehensive Logs & Top-N Spreadsheet)