"""
Data loader, sequence parser, and mapping utilities for the XES3G5M dataset.
"""
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

class SequenceParser:
    """Parses padded comma-separated sequences into structured arrays and masks."""

    def __init__(self, pad_val: int = -1, max_seq_len: int = 200):
        self.pad_val = pad_val
        self.max_seq_len = max_seq_len

    def parse_sequence_string(self, raw_str: Any) -> np.ndarray:
        """Parses comma-separated string into integer numpy array."""
        if pd.isna(raw_str):
            return np.array([], dtype=np.int64)
        items = [x.strip() for x in str(raw_str).split(",") if x.strip()]
        vals = []
        for it in items:
            if "_" in it:
                try:
                    vals.append(int(it.split("_")[0]))
                except ValueError:
                    continue
            else:
                try:
                    vals.append(int(it))
                except ValueError:
                    continue
        return np.array(vals, dtype=np.int64)

    def extract_active_mask(self, seq: np.ndarray) -> np.ndarray:
        """Returns boolean mask where True indicates non-padding real interactions."""
        return (seq != self.pad_val) & (seq >= 0)

class DatasetLoader:
    """Loads and validates XES3G5M interaction files and metadata."""

    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.parser = SequenceParser()
        self.questions_meta: Dict[str, Any] = {}
        self.kc_routes_zh: Dict[str, str] = {}
        self.kc_routes_en: Dict[str, str] = {}
        self.question_to_kcs: Dict[int, List[int]] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Loads question definitions and KC routes maps."""
        meta_dir = self.root / "metadata"
        q_path = meta_dir / "questions.json"
        if q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                self.questions_meta = json.load(f)

        zh_path = meta_dir / "kc_routes_map.json"
        if zh_path.exists():
            with open(zh_path, "r", encoding="utf-8") as f:
                self.kc_routes_zh = json.load(f)

        en_path = meta_dir / "kc_routes_map_en.json"
        if en_path.exists():
            with open(en_path, "r", encoding="utf-8") as f:
                self.kc_routes_en = json.load(f)

        # Build leaf KC name to KC ID mapping
        leaf_to_id = {v: int(k) for k, v in self.kc_routes_zh.items()}

        # Build Question to KC IDs mapping
        for qid_str, qdata in self.questions_meta.items():
            qid = int(qid_str)
            kc_ids = []
            for route in qdata.get("kc_routes", []):
                leaf = route.split("----")[-1].strip()
                if leaf in leaf_to_id:
                    kc_ids.append(leaf_to_id[leaf])
            self.question_to_kcs[qid] = kc_ids

    def get_kc_name(self, kc_id: int, lang: str = "en") -> str:
        """Returns English (or Chinese) name of KC."""
        str_id = str(kc_id)
        if lang == "en" and str_id in self.kc_routes_en:
            return self.kc_routes_en[str_id]
        if str_id in self.kc_routes_zh:
            return self.kc_routes_zh[str_id]
        return f"Concept_{kc_id}"

    def load_interaction_sequences(
        self,
        split: str = "train",
        val_fold: int = 4,
        max_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads interaction sequences from train_valid or test csv.
        Leakage-aware: if split=='train', fold != val_fold; if split=='val', fold == val_fold.
        If split=='test', loads test.csv.
        """
        if split in ["train", "val"]:
            csv_path = self.root / "kc_level" / "train_valid_sequences.csv"
        else:
            csv_path = self.root / "kc_level" / "test.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Sequences file not found: {csv_path}")

        records = []
        chunksize = 2000
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if split == "train":
                filtered = chunk[chunk["fold"] != val_fold]
            elif split == "val":
                filtered = chunk[chunk["fold"] == val_fold]
            else:
                filtered = chunk

            for _, row in filtered.iterrows():
                uid = int(row["uid"])
                q_arr = self.parser.parse_sequence_string(row["questions"])
                c_arr = self.parser.parse_sequence_string(row["concepts"])
                r_arr = self.parser.parse_sequence_string(row["responses"])

                # Valid sequence mask
                mask = self.parser.extract_active_mask(c_arr)
                if np.sum(mask) == 0:
                    continue

                records.append({
                    "uid": uid,
                    "questions": q_arr[mask],
                    "concepts": c_arr[mask],
                    "responses": r_arr[mask],
                    "length": int(np.sum(mask))
                })

                if max_rows and len(records) >= max_rows:
                    return records

        return records
