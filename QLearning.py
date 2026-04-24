import pandas as pd
import uuid
import hashlib

data = pd.DataFrame({
    'nombre': ['Ana', 'Juan', 'Luis'],
    'email': ['ana@ana.com', 'juan@juan.com', 'luis@luis.com'],
    'ubicacion': ['Ciudad A', 'Ciudad B', 'Ciudad C']
})

id_pseudo = []

for n in range(len(data)):
    id_pseudo.append(str(uuid.uuid4()))

print(id_pseudo)

data['id_pseudo'] = id_pseudo
data.drop('nombre', axis=1, inplace=True)
print(data)

def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

hash_emails = []

for email in data['email']:
    h_email = hash_data(email)
    hash_emails.append(h_email)
data['email'] = hash_emails

print(data)

mis_tokens = {}

def tokenizar(dato):
    token = str(uuid.uuid4())
    mis_tokens[token] = dato
    return token

def recuperar_dato(token):
    return mis_tokens.get(token, "Token no válido")

dato_original = "123-456-789"
token = tokenizar(dato_original)

print(f'Token generado: {token}')
print(f'Dato recuperado: {recuperar_dato(token)}')