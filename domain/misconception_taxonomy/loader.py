import yaml
from pathlib import Path
from typing import List, Dict

def load_taxonomy(path: str = "domain/misconception_taxonomy/taxonomy.yaml") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        nodes = yaml.safe_load(f)
    # basic validation
    valid_dims = {"completeness", "accuracy", "clarity", "coherence"}
    for node in nodes:
        assert node["dimension"] in valid_dims, f"Invalid dimension in node {node['id']}"
    return nodes

def get_nodes_by_dimension(nodes: List[Dict], dimension: str) -> List[Dict]:
    return [n for n in nodes if n["dimension"] == dimension]

def format_nodes_for_prompt(nodes: List[Dict]) -> str:
    return "\n".join(f"- {n['id']}: {n['description']}" for n in nodes)