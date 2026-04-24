import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

data = pd.DataFrame({
    'edad': ['22', '45', '36', '29', '55'],
    'género': ['F', 'M', 'M', 'F', 'M'],
    'contratado': [1, 0, 1, 0, 1]
})

agrupado = data.groupby('género')

for nombre, grupo in agrupado:
    print(nombre, grupo)

data_balanceado = pd.DataFrame()

for nombre, grupo in agrupado:
    grupo_balanceado = resample(grupo, 
                                replace=True,
                                n_samples=10,
                                random_state=123)
    data_balanceado = pd.concat([data_balanceado, grupo_balanceado])

print(data_balanceado)

