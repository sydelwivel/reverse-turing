# ai_mock.py
import numpy as np
from personas import _base_score, _clip

def ai_mock_score(resume_json):
    """
    Mock black-box scoring. 
    Has subtle correlations with prestige schools and brand employers.
    Swappable with a real model API call to return a numeric score (0–100).
    """
    base = _base_score(resume_json)

    # Safely extract school and employer
    school = resume_json.get("education", [{}])[0].get("school", "").lower()
    employer = resume_json.get("jobs", [{}])[0].get("employer", "").lower()

    # Prestige school boost
    prestige_bonus = 10 if any(k in school for k in
                                ["harvard", "stanford", "mit", "yale", "princeton"]) else 0

    # Big brand employer boost
    brand_bonus = 8 if any(k in employer for k in
                            ["google", "facebook", "amazon", "apple", "netflix", "microsoft", "meta"]) else 0

    # Small random noise to make it "black-boxy"
    noise = np.random.normal(0, 4)

    # Return final clipped score
    return _clip(base + prestige_bonus + brand_bonus + noise)
