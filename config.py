import yaml
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

TELEGRAM_TOKEN = config["telegram"]["token"]
