# data_generator.py
import random
import numpy as np
from models import normalize_resume, make_uid
import config

random.seed(config.RNG_SEED)
np.random.seed(config.RNG_SEED)

IVY_SCHOOLS = ["Harvard","Stanford","MIT","Yale","Princeton","Columbia","UPenn","Brown","Dartmouth","Cornell"]
FAANG_EMPLOYERS = ["Google","Facebook","Amazon","Apple","Netflix","Microsoft","Meta"]

SKILL_POOL = ["python","sql","java","react","nlp","computer vision","ml","docker","kubernetes","aws","data analysis"]

def generate_synthetic(n=200):
    rows = []
    for i in range(n):
        prestige = random.random() < 0.18
        school = random.choice(IVY_SCHOOLS) if prestige else f"State University {random.randint(1,200)}"
        employer = random.choice(FAANG_EMPLOYERS) if random.random() < 0.15 else f"Company{random.randint(1,500)}"
        gap_years = 0 if random.random() > 0.15 else random.choice([1,2,3])
        gender = random.choice(["male","female"])
        skills = random.sample(SKILL_POOL, k=random.randint(2,6))
        resume = {
            "uid": make_uid(),
            "name": "ANON",
            "education": [{"school": school, "degree": "BSc", "year": 2016}],
            "jobs": [{"employer": employer, "title": "Engineer", "start":"2017-01", "end":"2021-01"}],
            "skills": skills,
            "gender": gender,
            "gap_years": gap_years
        }
        rows.append(normalize_resume(resume))
    return rows
