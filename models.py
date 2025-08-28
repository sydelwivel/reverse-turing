# models.py
import uuid
from typing import Dict, Any

def make_uid():
    return str(uuid.uuid4())

def normalize_resume(resume: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(resume)
    if "uid" not in r:
        r["uid"] = make_uid()
    if "education" not in r:
        r["education"] = [{"school":"Unknown University", "degree":"BSc", "year":2016}]
    if "jobs" not in r:
        r["jobs"] = []
    if "skills" not in r:
        r["skills"] = []
    if "gender" not in r:
        r["gender"] = "unknown"
    if "gap_years" not in r:
        r["gap_years"] = _compute_gap_years(r.get("jobs", []))
    return r

def _compute_gap_years(jobs):
    # placeholder: for prototype return 0
    return 0
