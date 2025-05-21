import yaml, os
from functools import lru_cache

POLICY_PATH = os.getenv("FLASH_POLICY", "config/policy.yaml")

@lru_cache(maxsize=1)
def load_policy():
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) 