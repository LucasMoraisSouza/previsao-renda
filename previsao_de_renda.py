import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(layout = 'wide', 
                   page_title = 'Previsão de Renda',
                   page_icon = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTK1hzxWtM9Ic06iPNAIfLV6ELs3i6GOalCrQ&s')


st.write('# PREVISÃO DE RENDA')
st.write('---')
st.write('###### Esta é uma aplicação que tem o intuito de predizer a renda de uma observação.')
st.write('###### Ou seja: Qual a renda de um indivíduo lenvando em consideração algumas características específicas dele?')
st.write('###### Fazemos isso com auxilio de ferramentas e cálculos complexos envolvendo estatística e aprendizado de máquina.')
st.write('###### Para isso é preciso responder às perguntas que constam no formulário à esqueda da tela.')

st.write('----')




st.sidebar.write('***Estas são as perguntas que precisa responder. Pedimos para que ao final do formulário você clique no botão "Calcular renda" para que possamos realizar todos cálculos a fim de entregar-lhe a previsão num piscar de olhos!***')
st.sidebar.write('---')
st.sidebar.write('***Sobre o indivíduo:***')

opcoes_possui_veiculo = ['ESCOLHA UMA OPÇÃO','Sim', 'Não']
possui_veiculo = st.sidebar.selectbox("Possui um veículo?", opcoes_possui_veiculo)
if possui_veiculo == 'Sim':
    possui_veiculo = 1
elif possui_veiculo == 'Não':
    possui_veiculo = 0

opcoes_possui_imovel = ['ESCOLHA UMA OPÇÃO','Sim', 'Não']
possui_imovel = st.sidebar.selectbox("Possui um imóvel?", opcoes_possui_imovel)
if possui_imovel == 'Sim':
    possui_imovel = 1
elif possui_imovel == 'Não':
    possui_imovel = 0

opcoes_qtd_filhos = ['ESCOLHA UMA OPÇÃO', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
qtde_filhos = st.sidebar.selectbox("Quantos filhos possui?", opcoes_qtd_filhos)

opcoes_idade = ['ESCOLHA UMA OPÇÃO'] + [int(i) for i in range(1, 101)]
idade = st.sidebar.selectbox('Qual a idade?', opcoes_idade)

opcoes_tempo_emprego = ['ESCOLHA UMA OPÇÃO'] + [i/2 for i in range(1, 200)]
tempo_emprego = st.sidebar.selectbox('Quanto tempo de emprego aproximadamente? (em anos)', opcoes_tempo_emprego)

opcoes_qtde_pessoas_residencia = ['ESCOLHA UMA OPÇÃO'] + [int(i) for i in range(1, 21)]
qtde_pessoas_residencia = st.sidebar.selectbox('Quantas pessoas vivem na residência?', opcoes_qtde_pessoas_residencia)

opcoes_tipo_renda = ['ESCOLHA UMA OPÇÃO', 'Assalariado', 'Bolsista', 'Empresário' , 'Servidor público', 'Pensionista']
tipo_renda = st.sidebar.selectbox("Qual o tipo de renda?", opcoes_tipo_renda)

opcoes_nivel_educacao = ['ESCOLHA UMA OPÇÃO', 'Primário', 'Secundário', 'Superior incompleto', 'Superior completo', 'Pós graduação']
nivel_educacao = st.sidebar.selectbox("Qual o nível de educação/instrução?", opcoes_nivel_educacao)

opcoes_estado_civil = ['ESCOLHA UMA OPÇÃO', 'Solteiro', 'União', 'Casado', 'Separado', 'Viúvo']
estado_civil = st.sidebar.selectbox("Qual o estado civil?", opcoes_estado_civil)

opcoes_tipo_residencia = ['ESCOLHA UMA OPÇÃO', 'Casa', 'Com os pais', 'Governamental', 'Aluguel', 'Estúdio', 'Comunitário']
tipo_residencia = st.sidebar.selectbox('Qual o tipo de residência?', opcoes_tipo_residencia)

opcoes_sexo = ['ESCOLHA UMA OPÇÃO','M', 'F']
sexo = st.sidebar.selectbox("Qual o sexo?", opcoes_sexo)
if sexo == 'M':
    sexo = 1
elif sexo == 'F':
    sexo = 0

st.sidebar.write('---')

st.sidebar.write('***Clique no botão abaixo para calcular a renda ↓***')
botao = st.sidebar.button('Calcular renda')




if botao:

    st.write('Primeiramente gostaríamos de agradecer pelo preenchimento correto do formulário.') 
    st.write('Isso é muito importante para que consigamos fazer a previsão da melhor forma possível')
    st.write('---')
    

    renda = pd.read_csv(r"C:\Users\llluc\OneDrive\Área de Trabalho\Cursos Data Cience\Cientsta de dados\02 - CURSO DATA CIENTIST\Módulo 16 - Métodos de análise\Tarefas\Tarefa 01\projeto 2\input\previsao_de_renda.csv")


    lista = ['Unnamed: 0', 'data_ref', 'id_cliente']
    df = renda.drop(columns = lista)
    df = df.dropna()


    lista_var_categoricas = ['tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'sexo']
    df_dummie = pd.get_dummies(data = df, columns = lista_var_categoricas, drop_first = True, dummy_na=False)


    selecao = df_dummie.columns[df_dummie.dtypes == 'bool']
    df_dummie[selecao] = df_dummie[selecao].astype(int)


    df1 = df_dummie
    x = df1.drop('renda', axis = 1)
    y = df1['renda']


    x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y, test_size = 0.3, random_state = 100)


    rfr = RandomForestRegressor(n_estimators = 25, random_state = 100)
    rfr.fit(x_treinamento, y_treinamento)


    dicionario = {
        'posse_de_veiculo':[possui_veiculo],
        'posse_de_imovel':[possui_imovel],
        'qtd_filhos': [qtde_filhos],
        'idade': [idade],
        'tempo_emprego': [tempo_emprego],
        'qt_pessoas_residencia': [qtde_pessoas_residencia],

        'tipo_renda_Bolsista':[0],
        'tipo_renda_Empresário':[0],
        'tipo_renda_Pensionista':[0],
        'tipo_renda_Servidor público':[0],
        'tipo_renda_Assalariado':[0],


        'educacao_Pós graduação': [0],
        'educacao_Secundário': [0],
        'educacao_Superior completo': [0],
        'educacao_Superior incompleto': [0],
        'educacao_Primário': [0],

        'estado_civil_Separado': [0],
        'estado_civil_Solteiro': [0],
        'estado_civil_União': [0],
        'estado_civil_Viúvo': [0],
        'estado_civil_Casado': [0],

        'tipo_residencia_Casa': [0],
        'tipo_residencia_Com os pais': [0],
        'tipo_residencia_Comunitário': [0],
        'tipo_residencia_Estúdio': [0],
        'tipo_residencia_Governamental': [0],
        'tipo_residencia_Aluguel': [0],

        'sexo_M':[sexo]
    }




    if tipo_renda == 'Bolsista':
        dicionario['tipo_renda_Bolsista'] = [1]
    else:
        dicionario['tipo_renda_Bolsista'] = [0]

    if tipo_renda == 'Empresário':
        dicionario['tipo_renda_Empresário'] = [1]
    else:
        dicionario['tipo_renda_Empresário'] = [0]

    if tipo_renda == 'Pensionista':
        dicionario['tipo_renda_Pensionista'] = [1]
    else:
        dicionario['tipo_renda_Pensionista'] = [0]

    if tipo_renda == 'Servidor público':
        dicionario['tipo_renda_Servidor público'] = [1]
    else:
        dicionario['tipo_renda_Servidor público'] = [0]

    if tipo_renda == 'Assalariado':
        dicionario['tipo_renda_Assalariado'] = [1]
    else:
        dicionario['tipo_renda_Assalariado'] = [0]




    if nivel_educacao == 'Pós graduação':
        dicionario['educacao_Pós graduação'] = [1]
    else:
        dicionario['educacao_Pós graduação'] = [0]

    if nivel_educacao == 'Secundário':
        dicionario['educacao_Secundário'] = [1]
    else:
        dicionario['educacao_Secundário'] = [0]

    if nivel_educacao == 'Superior completo':
        dicionario['educacao_Superior completo'] = [1]
    else:
        dicionario['educacao_Superior completo'] = [0]

    if nivel_educacao == 'Superior incompleto':
        dicionario['educacao_Superior incompleto'] = [1]
    else:
        dicionario['educacao_Superior incompleto'] = [0]

    if nivel_educacao == 'Primário':
        dicionario['educacao_Primário'] = [1]
    else:
        dicionario['educacao_Primário'] = [0]



    if estado_civil == 'Separado':
        dicionario['estado_civil_Separado'] = [1]
    else:
        dicionario['estado_civil_Separado'] = [0]

    if estado_civil == 'Solteiro':
        dicionario['estado_civil_Solteiro'] = [1]
    else:
        dicionario['estado_civil_Solteiro'] = [0]

    if estado_civil == 'União':
        dicionario['estado_civil_União'] = [1]
    else:
        dicionario['estado_civil_União'] = [0]

    if estado_civil == 'Viúvo':
        dicionario['estado_civil_Viúvo'] = [1]
    else:
        dicionario['estado_civil_Viúvo'] = [0]

    if estado_civil == 'Casado':
        dicionario['estado_civil_Casado'] = [1]
    else:
        dicionario['estado_civil_Casado'] = [0]




    if tipo_residencia == 'Casa':
        dicionario['tipo_residencia_Casa'] = [1]
    else:
        dicionario['tipo_residencia_Casa'] = [0]

    if tipo_residencia == 'Com os pais':
        dicionario['tipo_residencia_Com os pais'] = [1]
    else:
        dicionario['tipo_residencia_Com os pais'] = [0]

    if tipo_residencia == 'Comunitário':
        dicionario['tipo_residencia_Comunitário'] = [1]
    else:
        dicionario['tipo_residencia_Comunitário'] = [0]

    if tipo_residencia == 'Estúdio':
        dicionario['tipo_residencia_Estúdio'] = [1]
    else:
        dicionario['tipo_residencia_Estúdio'] = [0]

    if tipo_residencia == 'Governamental':
        dicionario['tipo_residencia_Governamental'] = [1]
    else:
        dicionario['tipo_residencia_Governamental'] = [0]

    if tipo_residencia == 'Aluguel':
        dicionario['tipo_residencia_Aluguel'] = [1]
    else:
        dicionario['tipo_residencia_Aluguel'] = [0]

   
    df_previsao = pd.DataFrame(dicionario)
    df_previsao = df_previsao.drop(columns = ['tipo_renda_Assalariado', 'educacao_Primário', 'estado_civil_Casado', 'tipo_residencia_Aluguel'])

    predict = rfr.predict(df_previsao)

    #st.write(df_previsao)
    #st.write(x_teste)
    st.write(f' # Com base nas escolhas feitas no formulário, a renda desta observação gira em torno de R${np.round(predict, 2)[0]} reais.')
    st.write('---')
    st.write('###### NOTA: Esta é uma previsão feita com base em um banco de dados pré-existente.')
    st.write('###### Por este emotivo admite-se uma certa margem de erro para os resultados obtidos. \nCaso queira recalcular a renda mudando algum outro parâmetro do formulário, basta mudar a resposta e clicar novamente no botão "Calcular renda".')
    st.write('---')
