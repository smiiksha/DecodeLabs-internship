# tech_stack_recommender.py
# Project 3 – AI Recommendation Logic: Tech Stack Recommender
# DecodeLabs Industrial Training | Batch 2026
#
# Goal    : Map a user's skills/interests to the most relevant tech career paths
# Method  : Content-Based Filtering using TF-IDF Vectorization + Cosine Similarity
# Pipeline: Ingestion → Scoring → Sorting → Filtering  (4-step IPO model)

# -- Imports -----------------------------------------------------------
import math
import re


# ======================================================================
# SECTION 1 – DATASET  (raw_skills)
# ======================================================================
# Each entry represents a "job role item" in our recommendation engine.
# 'tags' are the skills/tools associated with that role.
# The recommender treats these like documents in a search index.

RAW_SKILLS = [
    {
        "role"  : "Data Scientist",
        "tags"  : ["python", "machine learning", "statistics", "sql",
                   "data analysis", "tensorflow", "pandas", "numpy",
                   "deep learning", "data visualization"]
    },
    {
        "role"  : "Machine Learning Engineer",
        "tags"  : ["python", "machine learning", "tensorflow", "pytorch",
                   "deep learning", "algorithms", "numpy", "model deployment",
                   "docker", "mlops"]
    },
    {
        "role"  : "Data Analyst",
        "tags"  : ["sql", "excel", "data analysis", "data visualization",
                   "python", "power bi", "tableau", "statistics",
                   "reporting", "pandas"]
    },
    {
        "role"  : "Backend Developer",
        "tags"  : ["python", "java", "sql", "apis", "rest",
                   "node.js", "databases", "server", "django",
                   "flask", "postgresql"]
    },
    {
        "role"  : "Frontend Developer",
        "tags"  : ["html", "css", "javascript", "react", "ui design",
                   "web development", "typescript", "figma",
                   "responsive design", "vue.js"]
    },
    {
        "role"  : "Full Stack Developer",
        "tags"  : ["html", "css", "javascript", "python", "react",
                   "node.js", "sql", "databases", "apis", "git",
                   "web development", "django"]
    },
    {
        "role"  : "DevOps Engineer",
        "tags"  : ["docker", "kubernetes", "aws", "ci/cd", "linux",
                   "automation", "git", "terraform", "cloud computing",
                   "bash", "monitoring"]
    },
    {
        "role"  : "Cloud Architect",
        "tags"  : ["aws", "azure", "google cloud", "cloud computing",
                   "kubernetes", "docker", "networking", "security",
                   "terraform", "automation"]
    },
    {
        "role"  : "Cybersecurity Analyst",
        "tags"  : ["networking", "security", "ethical hacking", "linux",
                   "python", "penetration testing", "cryptography",
                   "firewalls", "incident response", "siem"]
    },
    {
        "role"  : "AI Research Scientist",
        "tags"  : ["python", "deep learning", "machine learning", "pytorch",
                   "mathematics", "statistics", "algorithms", "nlp",
                   "computer vision", "research", "tensorflow"]
    },
    {
        "role"  : "NLP Engineer",
        "tags"  : ["python", "nlp", "transformers", "machine learning",
                   "tensorflow", "pytorch", "text processing", "bert",
                   "language models", "deep learning"]
    },
    {
        "role"  : "Computer Vision Engineer",
        "tags"  : ["python", "computer vision", "opencv", "deep learning",
                   "tensorflow", "pytorch", "image processing",
                   "convolutional neural networks", "object detection", "yolo"]
    },
    {
        "role"  : "Data Engineer",
        "tags"  : ["sql", "python", "etl", "spark", "hadoop",
                   "data pipelines", "databases", "aws", "kafka",
                   "airflow", "big data"]
    },
    {
        "role"  : "Mobile App Developer",
        "tags"  : ["java", "kotlin", "swift", "flutter", "react native",
                   "android", "ios", "mobile development", "ui design",
                   "apis"]
    },
    {
        "role"  : "System Administrator",
        "tags"  : ["linux", "networking", "automation", "bash",
                   "windows server", "security", "cloud computing",
                   "monitoring", "virtualization", "troubleshooting"]
    },
]


# ======================================================================
# SECTION 2 – TF-IDF VECTORIZER
# ======================================================================

def build_vocabulary(dataset):
    """
    Collect every unique skill tag across all job roles.
    This forms our shared 'vocabulary space' — the dimensions of our vectors.
    """
    vocab = set()
    for item in dataset:
        for tag in item["tags"]:
            vocab.add(tag.lower())
    # Sort for consistent indexing
    return sorted(vocab)


def compute_tf(tags):
    """
    Term Frequency: how often does each tag appear within this role's tag list?
    TF(t, d) = count(t in d) / total_terms(d)
    """
    total = len(tags)
    tf = {}
    for tag in tags:
        tag = tag.lower()
        tf[tag] = tf.get(tag, 0) + 1
    for tag in tf:
        tf[tag] = tf[tag] / total
    return tf


def compute_idf(dataset, vocabulary):
    """
    Inverse Document Frequency: penalises tags that appear in many roles.
    IDF(t) = log( total_documents / documents_containing_t )
    The log dampens the effect so common-but-not-universal terms
    are not over-penalised.
    """
    total_docs = len(dataset)
    idf = {}
    for term in vocabulary:
        docs_with_term = sum(
            1 for item in dataset
            if term in [t.lower() for t in item["tags"]]
        )
        # Smoothed IDF to avoid division-by-zero if term is in every doc
        idf[term] = math.log(total_docs / (1 + docs_with_term))
    return idf


def vectorize(tags, vocabulary, idf):
    """
    Convert a list of skill tags into a TF-IDF weighted numerical vector.
    The vector has one dimension for every term in the vocabulary.
    - Position i = tfidf weight of vocabulary[i]
    - Unused terms = 0.0
    """
    tf = compute_tf(tags)
    vector = []
    for term in vocabulary:
        weight = tf.get(term, 0.0) * idf.get(term, 0.0)
        vector.append(weight)
    return vector


# ======================================================================
# SECTION 3 – COSINE SIMILARITY ENGINE
# ======================================================================

def cosine_similarity(vec_a, vec_b):
    """
    Measures the angular alignment between two vectors.
    Invariant to magnitude — only the DIRECTION (preference orientation) matters.

    cos(θ) = (A · B) / (||A|| × ||B||)

    Returns:
        1.0  → perfectly aligned (identical interests)
        0.0  → orthogonal (no common interests)
       -1.0  → opposite (would not happen here since TF-IDF ≥ 0)
    """
    dot_product   = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a   = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b   = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        # Cold-start case: zero vector means no data → similarity undefined
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ======================================================================
# SECTION 4 – THE 4-STEP RECOMMENDATION PIPELINE
# ======================================================================

def recommend(user_skills, dataset, vocabulary, idf, top_n=3):
    """
    Executes the full IPO recommendation pipeline:

    Step 1 – INGESTION  : Convert user skill input to a TF-IDF vector.
    Step 2 – SCORING    : Calculate cosine similarity vs every job role.
    Step 3 – SORTING    : Rank roles by similarity score (descending).
    Step 4 – FILTERING  : Return only the Top-N results to prevent
                          'choice overload'.

    Args:
        user_skills (list[str]) : Skills the user provided (≥ 3)
        dataset (list[dict])    : The job-role knowledge base
        vocabulary (list[str])  : Shared feature space
        idf (dict)              : Pre-computed IDF weights
        top_n (int)             : Number of recommendations to return

    Returns:
        list of (role_name, score) tuples, highest score first
    """

    # -- Step 1: Ingestion ------------------------------------------------
    # Clean user input and build their personal TF-IDF vector
    clean_skills = [s.lower().strip() for s in user_skills]
    user_vector  = vectorize(clean_skills, vocabulary, idf)

    # -- Step 2: Scoring --------------------------------------------------
    scored = []
    for item in dataset:
        item_vector = vectorize(item["tags"], vocabulary, idf)
        score       = cosine_similarity(user_vector, item_vector)
        scored.append((item["role"], score))

    # -- Step 3: Sorting --------------------------------------------------
    scored.sort(key=lambda x: x[1], reverse=True)

    # -- Step 4: Filtering ------------------------------------------------
    return scored[:top_n]


# ======================================================================
# SECTION 5 – DISPLAY HELPERS
# ======================================================================

def print_banner():
    print('=' * 58)
    print('    Tech Stack Recommender — DecodeLabs AI Project 3')
    print('    Method : TF-IDF Vectorization + Cosine Similarity')
    print('    Track  : Content-Based Filtering (no ML model)')
    print('=' * 58)


def print_results(results, user_skills):
    print(f"\n{'─' * 58}")
    print(f"  Your Skills   : {', '.join(user_skills)}")
    print(f"{'─' * 58}")
    print(f"  TOP {len(results)} RECOMMENDED CAREER PATHS")
    print(f"{'─' * 58}")
    for rank, (role, score) in enumerate(results, start=1):
        bar_len  = int(score * 30)
        bar      = '█' * bar_len + '░' * (30 - bar_len)
        pct      = score * 100
        print(f"  #{rank}  {role:<30}  {pct:5.1f}%")
        print(f"       [{bar}]")
    print(f"{'─' * 58}\n")


# ======================================================================
# SECTION 6 – MAIN (Interactive Loop)
# ======================================================================

def get_user_skills():
    """
    Prompt user for ≥ 3 skills.
    Handles both comma-separated input and one-by-one entry.
    This is the 'Onboarding Survey' bypass for the Cold-Start problem.
    """
    print("\nEnter your skills / interests.")
    print("You can type them comma-separated, e.g.:")
    print("  python, machine learning, sql\n")

    raw = input("Your skills: ").strip()

    # Support comma or newline separated input
    skills = [s.strip() for s in re.split(r'[,\n]+', raw) if s.strip()]

    # Enforce minimum 3 skills for accurate cosine matching
    while len(skills) < 3:
        print(f"\n⚠  Only {len(skills)} skill(s) detected.")
        print("   At least 3 are required for accurate matching.")
        more = input("   Add more (comma-separated): ").strip()
        extra = [s.strip() for s in re.split(r'[,\n]+', more) if s.strip()]
        skills.extend(extra)

    return skills


def main():
    print_banner()

    # Pre-compute vocabulary and IDF once (shared across all queries)
    vocabulary = build_vocabulary(RAW_SKILLS)
    idf        = compute_idf(RAW_SKILLS, vocabulary)

    print(f"\n  Dataset  : {len(RAW_SKILLS)} job roles loaded")
    print(f"  Vocab    : {len(vocabulary)} unique skill terms indexed")

    while True:
        # ---- Get Input -----------------------------------------------
        user_skills = get_user_skills()

        # ---- Run Pipeline --------------------------------------------
        results = recommend(
            user_skills = user_skills,
            dataset     = RAW_SKILLS,
            vocabulary  = vocabulary,
            idf         = idf,
            top_n       = 3
        )

        # ---- Display Output ------------------------------------------
        print_results(results, user_skills)

        # ---- Cold Start Warning (no matches) -------------------------
        if all(score == 0.0 for _, score in results):
            print("  ⚠  No matching skills found in the dataset.")
            print("     Try keywords like: python, sql, machine learning,")
            print("     docker, javascript, deep learning, networking, etc.\n")

        # ---- Ask to Try Again ----------------------------------------
        again = input("  Try with different skills? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            print("\n  Exiting Tech Stack Recommender. Good luck with your career! 🚀")
            print('=' * 58)
            break
        print()


if __name__ == '__main__':
    main()
