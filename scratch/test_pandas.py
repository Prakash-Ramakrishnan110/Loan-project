import pandas as pd
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
try:
    res = df.astype(str).agg(' | '.join, axis=1)
    print("Agg worked")
    print(res)
except Exception as e:
    print(f"Agg failed: {e}")

try:
    res = df.astype(str).apply(' | '.join, axis=1)
    print("Apply worked")
    print(res)
except Exception as e:
    print(f"Apply failed: {e}")
