#libraries
import sklearn
import pandas as pd
from haversine import haversine
import plotly.express as px
from datetime import datetime

import folium
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import numpy as np

st.set_page_config(
    page_title="Visão Entregadores",
    page_icon="icone",
    layout = 'wide'
)

#################################################################################################
#FUNÇÕES
#################################################################################################
def clean_code(df):
    """Função de limpeza de codigo
        Tipos de limpeza:
        Remoção dos dados NaN
        Mudaça do tipo de daods de coluna
        Remoção de espacos das variáveis de texto
        Formatacao da coluna de dados
        Remocao do texto '(min)' da variável numerica de Time_taken
    
        Input = Dataframe
        Output = Dataframe
    
    """
    
    # 1. convertando a coluna Age de texto para numero
    #linhas_selecionadas = (df['Delivery_person_Age'] != 'NaN ')
    df = df.loc[df['Delivery_person_Age'] != 'NaN ', :].copy()
    linhas_selecionadas = (df['Road_traffic_density'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    
    #linhas_selecionadas = (df['City'] != 'NaN ')
    df = df.loc[df['City'] != 'NaN ', :].copy()
    linhas_selecionadas = (df['Festival'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    
    #limpeza time_taken
    #linhanan = (df['Time_taken(min)'] != 'NaN')
    df = df.loc[df['Time_taken(min)'] != 'NaN', :].copy()
    
    # Conversao de texto/categoria/string para numeros inteiros
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype( int )
    
    # 2. convertando a coluna Ratings de texto para numero decimal ( float )
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype( float )
    
    # 3. convertando a coluna order_date de texto para data
    df['Order_Date'] = pd.to_datetime( df['Order_Date'], format='%d-%m-%Y' )
    
    # 4. convertendo multiple_deliveries de texto para numero inteiro ( int )
    linhas_selecionadas = (df['multiple_deliveries'] != "NaN ")
    df = df.loc[linhas_selecionadas, :].copy()
    #df['multiple_deliveries'] = df['multiple_deliveries'].astype( int )
    
    # 6. Removendo os espacos dentro de strings/texto/object
    
    df.loc[:, 'ID'] = df.loc[:, 'ID'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:, 'Road_traffic_density'].str.strip()
    df.loc[:, 'Type_of_order'] = df.loc[:, 'Type_of_order'].str.strip()
    df.loc[:, 'Type_of_vehicle'] = df.loc[:, 'Type_of_vehicle'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Festival'] = df.loc[:, 'Festival'].str.strip()
     
    # 7. Limpando a coluna de time taken
    # Check if the value is a string before applying split
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: x.split('(min) ')[1] if isinstance(x, str) else np.nan)
    # Convert to int, handling NaN values
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(float).fillna(0).astype(int)
    return df
    
#-------------------------------------------------------------------------------------------
def top_delivers(df, top_asc):
    entregadores_rapidos = (df.loc[: ,['Delivery_person_ID','City','Time_taken(min)'] ]
                           .groupby(['City', 'Delivery_person_ID'])
                           .mean(numeric_only=top_asc)
                           .sort_values(['City', 'Time_taken(min)'], ascending=top_asc).reset_index() )
    
    ent_aux1 = entregadores_rapidos.loc[entregadores_rapidos['City'] == 'Metropolitan',:].head(10)
    ent_aux2 = entregadores_rapidos.loc[entregadores_rapidos['City'] == 'Urban',:].head(10)
    ent_aux3 = entregadores_rapidos.loc[entregadores_rapidos['City'] == 'Semi-Urban',:].head(10)
    entrega = pd.concat([ent_aux1, ent_aux2, ent_aux3]).reset_index ( drop=True )
    
    return entrega
    
############################################################################################
# INICIO DA ESTRUTURA DO CÓDIGO

#import dataset
df1 = pd.read_csv('dataset/train.csv')

#clean code
df = clean_code(df1)
pd.options.mode.copy_on_write = True


##########################################################################################
#Barra Lateral do streamlit
################################################################
image_path ='curry.png'
image = Image.open(image_path)
st.sidebar.image(image, width=120)


st.sidebar.title("Curry Company")
st.sidebar.markdown("### _O Delivery mais rápidico da Região_")

st.sidebar.markdown("""---""")

st.sidebar.markdown('## Selecione os Filtros:')


date_slider = st.sidebar.slider(
    'Selecione uma data:',
    value=datetime(2022, 4, 13),
    min_value=datetime(2022, 2, 11),
    max_value=datetime(2022, 6, 4),
    format='DD-MM-YYYY')

#st.header(date_slider)

st.sidebar.markdown("""---""")

traffic_options = st.sidebar.multiselect(
    ' Condições de Transito',
    ['Low', 'Medium', 'High', 'Jam'],
    default=['Low','Medium', 'High', 'Jam'] )

st.sidebar.markdown("""---""")
st.sidebar.markdown('Powerd by HLF.Meigabot')

#filtros
#filtro de data:
linhas_selecionadas = df['Order_Date'] < date_slider
df = df.loc[linhas_selecionadas, :]

#filtro de trânsito:
linhas_selecionadas = df['Road_traffic_density'].isin( traffic_options )
df = df.loc[linhas_selecionadas, :]

##########################################################################################
#layout no streamlit
##########################################################################################

st.title('Marketplace - Visão Entregadores')

##########################################################################################

tab1, tab2, tab3 = st.tabs( ['Visão Gerencial', '_', '_'] )

with tab1:
    with st.container():
        st.header( 'Metricas Gerais')
        col1, col2, col3, col4 = st.columns( 4, gap='large' )
        with col1:
            # A maior idade dos entregadores
            maior_idade = df.loc[:, 'Delivery_person_Age'].max()
            col1.metric( 'Maior de idade', maior_idade )
 
        with col2:
            # A menor idade dos entregadores
            menor_idade = df.loc[:, 'Delivery_person_Age'].min()
            col2.metric( 'Menor idade', menor_idade )
            
        with col3:
            # A Melhor Condição dos Veículos
            melhor_condicao = df.loc[:, 'Vehicle_condition'].max()
            col3.metric( 'Melhor condição de Veículo', melhor_condicao )
            
        with col4:
            # A Pior Condição dos Veículos
            pior_condicao = df.loc[:, 'Vehicle_condition'].min()
            col4.metric( 'Pior condição de Veículo', pior_condicao )
    
    with st.container():
        st.markdown( """---""" )
        st.header( 'Avaliações' )
        
        col1, col2 = st.columns( 2 )
        with col1:
            st.markdown( '##### Avalições Médias por Entregador' )
            df_avg_ratings_per_deliver = ( df.loc[:, ['Delivery_person_Ratings', 'Delivery_person_ID']]
                                              .groupby( 'Delivery_person_ID' )
                                              .mean()
                                              .reset_index() )
            st.dataframe( df_avg_ratings_per_deliver,  height=500 )
                
        with col2:
            st.markdown( '##### Avaliação Média por Intensidade de Tráfego' )
            df_avg_std_rating_by_traffic = ( df.loc[:, ['Delivery_person_Ratings', 'Road_traffic_density']]
                                               .groupby( 'Road_traffic_density')
                                               .agg( {'Delivery_person_Ratings': ['mean', 'std' ]} ) )
            # mudanca de nome das colunas
            df_avg_std_rating_by_traffic.columns = ['delivery_mean', 'delivery_std']
            # reset do index
            df_avg_std_rating_by_traffic = df_avg_std_rating_by_traffic.reset_index()
            st.dataframe( df_avg_std_rating_by_traffic )

            st.markdown( '##### Avaliação Média por Clima' )
            df_avg_std_rating_by_weather = ( df.loc[:, ['Delivery_person_Ratings', 'Weatherconditions']]
                                               .groupby( 'Weatherconditions')
                                               .agg( {'Delivery_person_Ratings': ['mean', 'std']} ) )

            # mudanca de nome das colunas
            df_avg_std_rating_by_weather.columns = ['delivery_mean', 'delivery_std']
            # reset do index
            df_avg_std_rating_by_weather = df_avg_std_rating_by_weather.reset_index()
            st.dataframe( df_avg_std_rating_by_weather )

    with st.container():
        st.markdown( """---""" )
        st.header( 'Velocidade de Entrega' )
        
        col1, col2 = st.columns( 2 )
        
        with col1:
            st.markdown( '#### Top Entregadores mais Rápidos' ) 
            entrega = top_delivers(df, top_asc=True)
            st.dataframe(entrega)

        with col2:
            st.markdown( '#### Top Entregadores mais Lentos' )
            entrega = top_delivers(df, top_asc=False)
            st.dataframe(entrega)



       











