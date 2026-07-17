from src.strategy.martingale import MartingaleStrategy

def test_martingale():
    mg = MartingaleStrategy()
    assert mg.get_next_stake(False) > 0
    assert mg.get_next_stake(True) == 2.0

print("Strategy tests passed.")
