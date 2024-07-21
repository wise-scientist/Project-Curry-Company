#libraries
import sklearn
import pandas as pd
from haversine import haversine
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import folium
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import numpy as np

st.set_page_config(
    page_title="Visão Restaurantes",
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

#----------------------------------------------------------------------------------------------------------------------

def distance(df, fig):
    if fig == False:
        cols =['Delivery_location_latitude','Delivery_location_longitude','Restaurant_latitude','Restaurant_longitude']
        df['distance'] = df.loc[:, cols].apply( lambda x: 
                    haversine( (x['Restaurant_latitude'], x['Restaurant_longitude']),
                               (x['Delivery_location_latitude'], x['Delivery_location_longitude']) ), axis=1)
        avg_distance = np.round(df['distance'].mean(), 2)
        return avg_distance
    else: 
        df['distance'] = df.loc[:,  ['Delivery_location_latitude','Delivery_location_longitude','Restaurant_latitude','Restaurant_longitude']].apply( lambda x: 
                        haversine( (x['Restaurant_latitude'], x['Restaurant_longitude']),
                                   (x['Delivery_location_latitude'], x['Delivery_location_longitude']) ), axis=1)
       
    avg_distance = df.loc[:, ['City', 'distance']].groupby('City').mean().reset_index()        
    fig = go.Figure(data=[ go.Pie( labels=avg_distance['City'], values=avg_distance['distance'], pull=[0.05, 0.1, 0])])
    return fig
        
#-------------------------------------------------------------------------------------------------------------
def avg_std_time_delivery(df, festival, op):
    """
    Esta função calcula a média ou desvio padrão do tempo médio de entrega:
    Prâmetros:
    Imput:
        - df = dataframe limpo
        - festival = booleano sobre o perído do evento Festival
            'Yes' = calcula para o período em que ocorre o Festival
            'No' = calcula para o período em que NÃO ocorre o Festival
        - op = tipo de operação ->
            'avg_time' = executa a média do tempo de entrega
            'std_time' = executa o desvio padrão do tempo de entrega
    Output: datafreme com duas colunas e uma linha
    """
    df_aux = (df.loc[:, ['Time_taken(min)', 'Festival']]
                .groupby( 'Festival' )
                .agg({'Time_taken(min)': ['mean','std']}) )
    
    df_aux.columns = ['avg_time','std_time']
    df_aux = df_aux.reset_index()
    
    df_aux = np.round(df_aux.loc[df_aux['Festival']==festival, op], 2)
    return df_aux
#------------------------------------------------------------------------------------------------------------
def sunburst_graph(df):
    df_aux = (df.loc[:, ['Time_taken(min)', 'City', 'Road_traffic_density']]
                .groupby( ['City', 'Road_traffic_density'] )
                .agg({'Time_taken(min)': ['mean','std']}) )
        
    df_aux.columns = ['avg_time','std_time']
    df_aux = df_aux.reset_index()
    fig = px.sunburst(df_aux, path=['City', 'Road_traffic_density'],      
                          values='avg_time', color='std_time', 
                          color_continuous_scale='RdBu', 
                          color_continuous_midpoint=np.average(df_aux['std_time']) )
    return fig
    
#----------------------------------------------------------------------------------------------------------------
def barr_graph(df):    
    df_aux = (df.loc[:, ['Time_taken(min)', 'City']]
                .groupby( 'City' )
                .agg({'Time_taken(min)': ['mean','std']}) )
            
    df_aux.columns = ['avg_time','std_time']
    df_aux = df_aux.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar( name='Control',
                          x=df_aux['City'],
                          y=df_aux['avg_time'],
                          error_y=dict( type='data', array=df_aux['std_time']) ) )
    fig.update_layout(barmode='group')                  
    return fig
    
############################################################################################
# INICIO DA ESTRUTURA DO CÓDIGO

#import dataset
df1 = pd.read_csv('dataset/train.csv')

df = clean_code(df1)
pd.options.mode.copy_on_write = True

##########################################################################################
#Barra Lateral no streamlit
###############################################################
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
 
#dataframe
#st.dataframe(df, use_container_width=True)


##########################################################################################
#layout no streamlit
##########################################################################################

st.title('Marketplace - Visão Restaurantes')

#########################################################################################

tab1, tab2, tab3 = st.tabs( ['Visão Gerencial', '_', '_'] )

with tab1: 
    with st.container():
        st.header("Métricas Gerais")
        col1, col2 = st.columns (2)
        
        with col1:
            delivery_unique = len(df.loc[:, 'Delivery_person_ID'].unique() )
            col1.metric('Quantidade de Entregadores', delivery_unique)
        
        with col2:
            avg_distance = distance(df, fig=False)
            col2.metric('Distância Média entre Restaurantes e Entregas', avg_distance)
            
            
    with st.container():
        st.header('Métricas com Festival')
        col3, col4 = st.columns (2)

        with col3:
            df_aux = avg_std_time_delivery(df,'Yes','avg_time')
            col3.metric('Tempo Médio de Entrega no Festival', df_aux)
        
        with col4:
            df_aux = avg_std_time_delivery(df,'Yes','std_time')
            col4.metric('Variação do tempo de entrega no Festival', df_aux)

    with st.container():
        st.header('Métricas sem Festival')
        col5, col6 = st.columns (2)

        with col5:
            df_aux = avg_std_time_delivery(df,'No','avg_time')
            col5.metric('Tempo Médio de Entrega sem Festival', df_aux)
            
        with col6:
            df_aux = avg_std_time_delivery(df,'No','std_time')
            col6.metric('Variação do tempo de entrega sem Festival', df_aux)

    col1, col2 = st.columns (2)
    
    with col1:
        with st.container():
            st.markdown('### Tempo Médio por distância entre Restaurante e Região')
            fig = distance(df, True)
            st.plotly_chart( fig, use_container_width=True )

    with col2:        
        with st.container():
            st.markdown('### Tempo Médio por Região e Intensidade de Tráfego')       
            fig = sunburst_graph(df)
            st.plotly_chart(fig,use_container_width=True)
        
    with st.container():
        st.header("Distribuição e Variabilidade do Tempo de Entrega por Região")
        fig = barr_graph(df)
        st.plotly_chart(fig)              
        
    with st.container():
        st.header("Distribuição da distância")
        df_aux = (df.loc[:, ['Time_taken(min)', 'City', 'Type_of_order']]
                    .groupby( ['City', 'Type_of_order'] )
                    .agg({'Time_taken(min)': ['mean','std']}) )
            
        df_aux.columns = ['avg_time','std_time']
        df_aux = df_aux.reset_index()
        st.dataframe(df_aux, use_container_width=True)








