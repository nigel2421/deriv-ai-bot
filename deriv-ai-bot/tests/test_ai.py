from src.ai.predictor import Predictor

def test_predictor():
    pred = Predictor()
    result = pred.predict([{'quote': 1234.5}]*60)
    assert 'digit' in result
    assert 0 <= result['confidence'] <= 1

print("AI tests passed.")
