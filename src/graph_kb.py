import json
from pathlib import Path
from typing import Dict, Any

import networkx as nx


class KnowledgeGraph:
    """
    Small knowledge graph used by the PSC Evidence Assistant.

    The graph stores:

        entities     -> nodes
        relationships -> directed edges

    Example:

        PSC
         |
         | ASSOCIATED_WITH
         v
        Macrophages

    Every relationship can also remember which document
    chunk provided the evidence.
    """

    def __init__(self):
        """
        Create an empty directed multi-graph.

        MultiDiGraph means:

        - directed:
            A -> B is different from B -> A

        - multi:
            more than one relationship can exist
            between the same two entities
        """

        self.g = nx.MultiDiGraph()

    # =========================================================
    # ADD ENTITY
    # =========================================================

    def add_entity(
        self,
        name: str,
        etype: str = "Other",
        normalized_id: str | None = None,
    ) -> None:
        """
        Add an entity to the graph.

        Example:

            add_entity(
                name="PSC",
                etype="Disease"
            )

        creates a graph node representing PSC.
        """

        name = (name or "").strip()
        etype = (etype or "Other").strip()

        if not name:
            return

        if normalized_id is not None:
            normalized_id = normalized_id.strip() or None

        # -----------------------------------------------------
        # New entity
        # -----------------------------------------------------

        if not self.g.has_node(name):

            self.g.add_node(
                name,
                type=etype,
                normalized_id=normalized_id,
            )

            return

        # -----------------------------------------------------
        # Existing entity
        #
        # Do not duplicate it.
        # Instead, update missing metadata when possible.
        # -----------------------------------------------------

        existing = self.g.nodes[name]

        if (
            not existing.get("type")
            or existing.get("type") == "Other"
        ):
            existing["type"] = etype

        if (
            not existing.get("normalized_id")
            and normalized_id
        ):
            existing["normalized_id"] = normalized_id

    # =========================================================
    # ADD RELATION
    # =========================================================

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        evidence_chunk_id: str,
    ) -> None:
        """
        Add a directed relationship between two entities.

        Example:

            add_relation(
                source="Macrophages",
                relation="ASSOCIATED_WITH",
                target="PSC",
                evidence_chunk_id="abc123_0012"
            )

        produces:

            Macrophages
                |
                | ASSOCIATED_WITH
                v
               PSC

        and stores abc123_0012 as evidence.
        """

        source = (source or "").strip()
        target = (target or "").strip()
        relation = (relation or "").strip()
        evidence_chunk_id = (
            evidence_chunk_id or ""
        ).strip()

        # A relationship without source or target
        # is not useful.
        if not source or not target:
            return

        # A relationship type is required.
        if not relation:
            return

        # Evidence is important for this project.
        #
        # We do not want unsupported graph relationships.
        if not evidence_chunk_id:
            return

        # -----------------------------------------------------
        # Make sure both nodes exist.
        #
        # Usually agent.py has already added the entities,
        # but this prevents NetworkX from silently creating
        # nodes without useful metadata.
        # -----------------------------------------------------

        if not self.g.has_node(source):

            self.g.add_node(
                source,
                type="Other",
                normalized_id=None,
            )

        if not self.g.has_node(target):

            self.g.add_node(
                target,
                type="Other",
                normalized_id=None,
            )

        # -----------------------------------------------------
        # Create a stable key for this relationship.
        #
        # Because we use MultiDiGraph, multiple edges can
        # exist between the same source and target.
        # -----------------------------------------------------

        edge_key = (
            f"{relation}:{evidence_chunk_id}"
        )

        # Avoid adding the exact same evidence relation
        # more than once.
        if self.g.has_edge(
            source,
            target,
            key=edge_key,
        ):
            return

        self.g.add_edge(
            source,
            target,
            key=edge_key,
            relation=relation,
            evidence_chunk_id=evidence_chunk_id,
        )

    # =========================================================
    # QUERY ONE ENTITY
    # =========================================================

    def query_neighbors(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """
        Return relationships starting from an entity.

        Example:

            query_neighbors("Macrophages")

        may return:

            {
                "node": "Macrophages",
                "found": True,
                "neighbors": [
                    {
                        "to": "PSC",
                        "relation": "ASSOCIATED_WITH",
                        "evidence_chunk_id": "abc123_0012"
                    }
                ]
            }
        """

        name = (name or "").strip()

        if not name:
            return {
                "node": name,
                "found": False,
                "neighbors": [],
            }

        if not self.g.has_node(name):

            return {
                "node": name,
                "found": False,
                "neighbors": [],
            }

        neighbors = []

        for (
            source,
            target,
            key,
            data,
        ) in self.g.out_edges(
            name,
            keys=True,
            data=True,
        ):

            neighbors.append(
                {
                    "to": target,
                    "key": key,
                    **data,
                }
            )

        return {
            "node": name,
            "found": True,
            "type": self.g.nodes[name].get(
                "type",
                "Other",
            ),
            "normalized_id": self.g.nodes[name].get(
                "normalized_id"
            ),
            "neighbors": neighbors,
        }

    # =========================================================
    # BASIC GRAPH INFORMATION
    # =========================================================

    def stats(self) -> Dict[str, int]:
        """
        Return basic information about the graph.

        Example:

            {
                "nodes": 12,
                "edges": 18
            }
        """

        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
        }

    # =========================================================
    # SAVE GRAPHML
    # =========================================================

    def save_graphml(
        self,
        path: Path,
    ) -> None:
        """
        Save the graph in GraphML format.

        GraphML can later be opened by graph/network tools.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # GraphML does not handle None values well.
        #
        # Make a copy and replace None with empty strings.
        graph_copy = self.g.copy()

        for _, attributes in graph_copy.nodes(
            data=True
        ):

            for key, value in list(
                attributes.items()
            ):

                if value is None:
                    attributes[key] = ""

        for _, _, _, attributes in graph_copy.edges(
            keys=True,
            data=True,
        ):

            for key, value in list(
                attributes.items()
            ):

                if value is None:
                    attributes[key] = ""

        nx.write_graphml(
            graph_copy,
            path,
        )

    # =========================================================
    # SAVE JSON
    # =========================================================

    def save_json(
        self,
        path: Path,
    ) -> None:
        """
        Save the graph as human-readable JSON.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "nodes": [
                {
                    "id": node,
                    **self.g.nodes[node],
                }
                for node in self.g.nodes()
            ],

            "edges": [
                {
                    "source": source,
                    "target": target,
                    "key": key,
                    **edge_data,
                }
                for (
                    source,
                    target,
                    key,
                    edge_data,
                ) in self.g.edges(
                    keys=True,
                    data=True,
                )
            ],
        }

        path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
