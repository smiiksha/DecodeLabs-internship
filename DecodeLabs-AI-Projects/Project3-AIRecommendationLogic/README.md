# Project 3 – AI Recommendation Logic: Tech Stack Recommender

## Overview
A content-based filtering recommendation engine that maps a user's raw skills and interests to the most relevant tech career paths. Built entirely from scratch — no ML libraries — using **TF-IDF Vectorization** and **Cosine Similarity** implemented with pure Python math.

## Objective
Implement a full recommendation pipeline following the **IPO (Input–Process–Output) model**:
1. Ingest user skill input (≥ 3 skills) → build a TF-IDF vector
2. Score every job role against the user profile using Cosine Similarity
3. Sort all roles in descending order by similarity score
4. Filter and return the Top-3 most relevant career paths

## Files

| File | Description |
|------|-------------|
| `tech_stack_recommender.py` | Main implementation — full pipeline, dataset, TF-IDF engine, cosine similarity, interactive loop |
| `output_run.txt` | Saved terminal output from 5 different test runs |
| `README.md` | This file |
| `screenshots/` | Terminal screenshots (see submission) |

## How to Run

```bash
# No external libraries needed — pure Python 3 only!
python tech_stack_recommender.py
```

Then enter your skills when prompted:
```
Your skills: python, machine learning, sql
```

The script will ask if you want to try again with different skills after each result.

## Dataset

15 job roles, each with 10 associated skill tags, covering:

| Role | Key Skills |
|------|------------|
| Data Scientist | Python, ML, SQL, Statistics |
| Machine Learning Engineer | Python, TensorFlow, PyTorch, MLOps |
| Data Analyst | SQL, Excel, Power BI, Tableau |
| Backend Developer | Python, Java, APIs, Databases |
| Frontend Developer | HTML, CSS, JavaScript, React |
| Full Stack Developer | HTML, JS, Python, Databases, APIs |
| DevOps Engineer | Docker, Kubernetes, AWS, CI/CD |
| Cloud Architect | AWS, Azure, GCP, Terraform |
| Cybersecurity Analyst | Networking, Linux, Ethical Hacking |
| AI Research Scientist | Deep Learning, NLP, Mathematics |
| NLP Engineer | Transformers, BERT, Language Models |
| Computer Vision Engineer | OpenCV, CNNs, Object Detection |
| Data Engineer | ETL, Spark, Kafka, Airflow |
| Mobile App Developer | Flutter, React Native, Android/iOS |
| System Administrator | Linux, Networking, Automation |

## Algorithm: How It Works

### Why Content-Based over Collaborative Filtering?
Content-based filtering maps user preferences **directly to item attributes** — no historical user interaction data needed. This makes it immediately deployable without a massive dataset.

### Step-by-step:

**1. Vector Mapping (TF-IDF)**
- Build a shared vocabulary of all 88 unique skill terms
- For each job role and the user's profile, compute a TF-IDF weighted vector
- TF (Term Frequency): how often a skill appears in a role's tag list
- IDF (Inverse Document Frequency): penalises generic skills that appear in many roles (e.g., "python") — rewards specific ones (e.g., "mlops", "yolo")

**2. Cosine Similarity**
- Measures the angular alignment between two vectors
- Invariant to magnitude — only the *direction* of interests matters
- Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
- Score 1.0 = perfect match | Score 0.0 = no overlap

**3. Ranking Pipeline**
```
User Input → TF-IDF Vector → Cosine Score (×15 roles) → Sort → Top-3 Output
```

## Sample Results

| Input Skills | #1 Match | #2 Match | #3 Match |
|---|---|---|---|
| python, machine learning, sql | Data Scientist 36.7% | ML Engineer 20.3% | AI Research Scientist 19.1% |
| docker, kubernetes, aws | Cloud Architect 50.6% | DevOps Engineer 48.6% | ML Engineer 16.1% |
| javascript, react, html, css | Full Stack Dev 64.2% | Frontend Dev 57.1% | — |
| deep learning, pytorch, nlp, transformers | NLP Engineer 61.5% | AI Research Scientist 33.7% | ML Engineer 15.8% |
| networking, linux, security | Sys Admin 44.8% | Cybersecurity Analyst 42.0% | Cloud Architect 31.3% |

## Key Concepts Used

- **Content-Based Filtering** — item attributes drive recommendations
- **TF-IDF Weighting** — penalises generic tags, rewards specific ones
- **Cosine Similarity** — magnitude-invariant preference alignment
- **Cold-Start Bypass** — onboarding survey (enforcing ≥ 3 inputs) bootstraps the user vector
- **Top-N Filtering** — prevents choice overload by returning only 3 results
- **Pure Python implementation** — no scikit-learn, no external dependencies

## Tech Stack

- Python 3.x (standard library only: `math`, `re`)
- No pip installs required

## Concepts Demonstrated (as per Project 3 spec)

✅ Takes user input (≥ 3 skills/interests)  
✅ Matches preferences using mathematical similarity logic (TF-IDF + Cosine)  
✅ Displays recommended items with match percentage and visual bar  
✅ Logic building, pattern matching, recommendation concepts  
✅ Cold-start problem awareness and bypass strategy  
