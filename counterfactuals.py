# counterfactuals.py
import copy
import random

def generate_counterfactuals(resumes, n_pairs=5):
    """
    Generate counterfactual resume pairs by minimally editing sensitive attributes.
    Each pair = (original, counterfactual).
    """
    pairs = []
    sensitive_attributes = ["gender", "education_school", "gap_years"]

    for resume in random.sample(resumes, min(n_pairs, len(resumes))):
        # Copy resume for counterfactual
        cf = copy.deepcopy(resume)

        # Toggle gender
        if "gender" in cf:
            cf["gender"] = "female" if cf["gender"].lower() == "male" else "male"

        # Modify school prestige (swap Ivy <-> Non-Ivy)
        if "education_school" in cf:
            school = cf["education_school"].lower()
            if any(ivy in school for ivy in ["harvard", "yale", "princeton", "columbia", "brown", "dartmouth", "upenn", "cornell"]):
                cf["education_school"] = "Generic State University"
            else:
                cf["education_school"] = "Harvard University"

        # Adjust gap years (toggle 0 ↔ 3 for effect)
        if "gap_years" in cf:
            cf["gap_years"] = 0 if cf["gap_years"] > 0 else 3

        pairs.append((resume, cf))

    return pairs
