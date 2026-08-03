import json
from pathlib import Path
from typing import Dict

import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_entity(self, name: str, etype: str, normalized_id: str | None = None):
        name = (name or "").strip()
        if not name:
            return
        if not self.g.has_node(name):
            self.g.add_node(name, type=etype, normalized_id=normalized_id)

    def add_relation(self, source: str, relation: str, target: str, evidence_chunk_id: str):
        source = (source or "").strip()
        target = (target or "").strip()
        if not source or not target:
            return
        self.g.add_edge(
            source,
            target,
            key=f"{relation}:{evidence_chunk_id}",
            relation=relation,
            evidence_chunk_id=evidence_chunk_id,
        )

    def query_neighbors(self, name: str) -> Dict:
        if not self.g.has_node(name):
            return {"node": name, "found": False, "neighbors": []}
        neighbors = []
        for u, v, k, data in self.g.edges(name, keys=True, data=True):
            neighbors.append({"to": v, **data})
        return {"node": name, "found": True, "neighbors": neighbors}

    def save_graphml(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.g, path)

    def save_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [{"id": n, **self.g.nodes[n]} for n in self.g.nodes()],
            "edges": [
                {"source": u, "target": v, "key": k, **d}
                for u, v, k, d in self.g.edges(keys=True, data=True)
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
