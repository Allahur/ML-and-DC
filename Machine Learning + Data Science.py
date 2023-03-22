# Nota: Utilize Jupyter ou Google Colab
# Nota: Baixe os arquivos postados neste repositório.

# Passo a Passo

# 1. Entendimento do Desafio

# 2. Entendimento da Empresa/Área   
    # Prever o preço de um barco baseado nas características dele; ano, material, usado/novo, etc.

# 3. Extração/Obtenção de Dados

import pandas as pd
  
tabela = pd.read_csv("barcos_ref.csv")
print(tabela) 

# 4. Ajustes de Dados(Limpeza de Dados)

print(tabela.info())

# 5. Análise Exploratória 
   # Correlação entre as informações da base de dados 
correlacao = tabela.corr()[["Preco"]]

import seaborn as sb 
import matplotlib.pyplot as plt
# Crie o gráfico 
sb.heatmap(correlacao, cmap = "crest", annot = True)
# Exiba o gráfico
plt.show()

# 6. Modelagem + Algoritmos 
  # Dividir a base em x e y 

y = tabela["Preco"]
# axis = 0 -> linhas, axis = 1 -> colunas
x = tabela.drop("Preco", axis = 1)

# train text split
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 1)

# Importar a inteligência artificial 
  # Regressão Linear e Arvore de Decisão 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 

# Cria a inteligência artificial
model_regressionlinear = LinearRegression()
model_decisiontree = RandomForestRegressor() 

# Treinar a inteligência artificial 

model_regressionlinear.fit(x_treino, y_treino)
model_decisiontree.fit(x_treino, y_treino)


# 7. Interpretação do Resultado 

# Escolher o melhor modelo -> Rª fazer novas previsão
from sklearn.metrics import r2_score

previsao_rglinear = model_regressionlinear.predict(x_teste) 

previsao_treedecision = model_decisiontree.predict(x_teste)

print(r2_score(y_teste, previsao_rglinear()))
print(r2_score(y_teste, previsao_treedecision()))

# Visualizar as Previsões 

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Arvore de Decisão"] = previsao_treedecision
tabela_auxiliar["Regressão Linear"] = previsao_rglinear

sb.lineplot(data=tabela_auxiliar)
plt.show()


# Fazer novas previsões (Usando a inteligência artificial na prática)
tabela_nova = pd.read_csv("novos_barcos.csv")
print(tabela_nova)

previsao = model_decisiontree.predict(tabela_nova) 
print(previsao) 
