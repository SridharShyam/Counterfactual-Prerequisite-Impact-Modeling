import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

# Set OpenMP environment variable defensively
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_project_root() -> Path:
    """Returns the absolute root path of the CFLP_Project repository."""
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads YAML configuration with relative paths resolved to absolute paths.
    Automatically loads environment variables from .env and applies overrides
    for server ports, hostnames, dataset locations, and environment settings.
    """
    root = get_project_root()
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    if config_path is None:
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow environment variables to override sensitive or environment-specific paths
    env_dataset_root = os.getenv("CFLP_DATA_DIR")
    if env_dataset_root:
        cfg["data"]["dataset_root"] = env_dataset_root

    # Resolve data paths relative to project root or dataset root
    ds_root = Path(cfg["data"]["dataset_root"])
    if not ds_root.is_absolute():
        ds_root = (root / ds_root).resolve()
    cfg["data"]["dataset_root_abs"] = str(ds_root)

    return cfg
