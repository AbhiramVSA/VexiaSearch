from src.rag import rrf

def test_rrf():
    ranks = [['a','b','c'], ['b','c','d']]
    fused = rrf(ranks)
    assert fused[0] in {'b','c'}
