from bayesmixpy import build_bayesmix

def test_build():
    success = build_bayesmix()
    assert success == True
