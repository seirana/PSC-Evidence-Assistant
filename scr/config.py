from dataclasses import dataclass
from pathlib import Path

'''
Without @dataclass:

class Settings:
    def __init__(self, chunk_size, top_k):
        self.chunk_size = chunk_size
        self.top_k = top_k


With @dataclass:

from dataclasses import dataclass

@dataclass
class Settings:
    chunk_size: int
    top_k: int
    
(frozen=True)    
This prevents accidental modification — great for configs.    
'''

@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    corpus_dir: Path = project_root / "data" / "corpus"
    outputs_dir: Path = project_root / "outputs"
    demos_dir: Path = outputs_dir / "demos"

    # RAG settings
    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 250
    top_k: int = 6

    # Graph settings
    graph_path_graphml: Path = outputs_dir / "knowledge_graph.graphml"
    graph_path_json: Path = outputs_dir / "knowledge_graph.json"
