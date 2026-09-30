import sys
print("Python:", sys.executable)
import torch
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
import networkx as nx
print("NetworkX:", nx.__version__)
import fastapi
print("FastAPI:", fastapi.__version__)
