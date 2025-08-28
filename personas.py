# personas.py

# Each persona is a scoring function: takes resume dict → outputs numeric score

def ivy_only_bias(resume):
    """Favor Ivy League schools, penalize others."""
    school = resume.get("education_school", "").lower()
    ivy_league = ["harvard", "yale", "princeton", "columbia", "brown",
                  "dartmouth", "upenn", "cornell"]
    if any(ivy in school for ivy in ivy_league):
        return 0.9
    return 0.5

def gap_year_penalty(resume):
    """Penalize candidates with long gap years."""
    gap = resume.get("gap_years", 0)
    return max(0.9 - 0.1 * gap, 0.1)

def brand_snob_bias(resume):
    """Favor FAANG employers."""
    employers = str(resume.get("jobs_employer", "")).lower()
    faang = ["google", "amazon", "facebook", "meta", "apple", "microsoft", "netflix"]
    if any(b in employers for b in faang):
        return 0.9
    return 0.5

def gender_penalty_bias(resume):
    """Research-only persona: penalize female resumes."""
    gender = resume.get("gender", "").lower()
    if gender == "female":
        return 0.4
    return 0.8

# Dictionary mapping persona names → function
bias_personas = {
    "Ivy-only Bias": ivy_only_bias,
    "Gap-year Penalty": gap_year_penalty,
    "Brand-snob Bias": brand_snob_bias,
    "Gender Penalty": gender_penalty_bias,
}
# personas.py

"""
Each persona is a scoring function: takes a resume dict → outputs numeric score (0–1).
"""

def _base_score(resume):
    """
    Provide a baseline score for any resume.
    Implementation: weighted default baseline.
    Scales roughly into a 0–100 range (before clipping).
    """
    # Start with a neutral score around 50
    score = 50.0
    
    # Simple heuristic: add minor boosts if resume has some info
    if resume.get("education"):
        score += 5
    if resume.get("jobs"):
        score += 5
    if resume.get("skills"):
        score += 5
    
    return score


def _clip(val, low=0, high=100):
    """
    Ensure the score stays within bounds [0, 100].
    """
    return max(low, min(high, val))


def ivy_only_bias(resume):
    """Favor Ivy League schools, penalize others."""
    school = resume.get("education_school", "").lower()
    ivy_league = ["harvard", "yale", "princeton", "columbia", "brown",
                  "dartmouth", "upenn", "cornell"]
    if any(ivy in school for ivy in ivy_league):
        return 0.9
    return 0.5


def gap_year_penalty(resume):
    """Penalize candidates with long gap years."""
    gap = resume.get("gap_years", 0)
    return max(0.9 - 0.1 * gap, 0.1)


def brand_snob_bias(resume):
    """Favor FAANG employers."""
    employers = str(resume.get("jobs_employer", "")).lower()
    faang = ["google", "amazon", "facebook", "meta", "apple", "microsoft", "netflix"]
    if any(b in employers for b in faang):
        return 0.9
    return 0.5


def gender_penalty_bias(resume):
    """(For research) Penalize female resumes artificially."""
    gender = resume.get("gender", "").lower()
    if gender == "female":
        return 0.4
    return 0.8


# Dictionary mapping persona names → function
bias_personas = {
    "Ivy-only Bias": ivy_only_bias,
    "Gap-year Penalty": gap_year_penalty,
    "Brand-snob Bias": brand_snob_bias,
    "Gender Penalty": gender_penalty_bias,
}
