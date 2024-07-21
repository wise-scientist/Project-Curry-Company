#libraries
import sklearn
import pandas as pd
from haversine import haversine
import plotly.express as px
from datetime import datetime
#import plotly.graph_objrcts as go

import folium
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import numpy as np

st.set_page_config(
    page_title="Visão Empresa",
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
    linhas_selecionadas = (df['Delivery_person_Age'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    
    linhas_selecionadas = (df['Road_traffic_density'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    
    #linhas_selecionadas = (df['City'] != 'NaN ')
    df = df.loc[df['City'] != 'NaN ', :].copy()
    
    linhas_selecionadas = (df['Festival'] != 'NaN ')
    df = df.loc[linhas_selecionadas, :].copy()
    
    #limpeza time_taken
    linhanan = (df['Time_taken(min)'] != 'NaN')
    df = df.loc[linhanan, :].copy()
    
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
    
    # Remover spaco da string
    #for i in range( len( df ) ):
     #   df.loc[i, 'ID'] = df.loc[i, 'ID'].strip()
      #  df.loc[i, 'Delivery_person_ID'] = df.loc[i, 'Delivery_person_ID'].strip()
    
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
#-------------------------------------------------------------------------------

def order_day(df):
    df.loc[:,['ID', 'Order_Date']].groupby('Order_Date').count().reset_index()
    fig = px.bar(df, x='Order_Date', y='ID')
    return fig
#---------------------------------------------------------------------------------------

def traffic_order_share(df):
        df_aux3 = df.loc[:,['ID', 'Road_traffic_density']].groupby('Road_traffic_density').count().reset_index()
        #limpeza do NaN
        df_aux3 = df_aux3.loc[df_aux3['Road_traffic_density'] != "NaN ", :]
        #Criar coluna % fazendo a conta
        df_aux3['porcent'] = df_aux3['ID'] / df_aux3['ID'].sum()
        df_aux3.head()
        #gáfico
        fig = px.pie(df_aux3, values='porcent', names='Road_traffic_density')
        return fig

#---------------------------------------------------------------------------------

def traffic_order_city(df):
    df_aux4 = df.loc[:,['ID', 'City', 'Road_traffic_density']].groupby(['City', 'Road_traffic_density']).count().reset_index()
    df_aux4.head()
    #limpeza do NaN
    df_aux4 = df_aux4.loc[df_aux4['Road_traffic_density'] != "NaN ", :]
    df_aux4 = df_aux4.loc[df_aux4['City'] != "NaN ", :]
    df_aux4.head()
    fig = px.scatter(df_aux4, x='City', y='Road_traffic_density', size='ID', color='City')
    return fig

#---------------------------------------------------------------------------------

def order_week(df):
    #criar a coluna week_year
    df['week_year'] = df['Order_Date'].dt.strftime( '%U')
    df_aux2 = df.loc[:,['ID', 'week_year']].groupby('week_year').count().reset_index()
    fig = px.line(df_aux2, x='week_year', y='ID')
    return fig

#----------------------------------------------------------------------------------

def order_share_week(df):
    #criar df com semana do ano por quantidade de entrega
    df_aux5a = df.loc[: ,['ID', 'week_year']].groupby('week_year').count().reset_index()
    #criar df com quantidade de entregadores únicos por semana
    df_aux5b = df.loc[: ,['Delivery_person_ID', 'week_year']].groupby('week_year').nunique().reset_index()
    #juntar todas as colunas em um df
    df_aux5 = pd.merge(df_aux5a, df_aux5b, how='inner')
    #crair uma coluna: quantidade de pedidos / quantidade de entregadores únicos
    df_aux5['ID/uni_Delivery_person_ID'] = df_aux5['ID'] / df_aux5['Delivery_person_ID']
    #grafico de linhas
    fig = px.line(df_aux5, x='week_year', y='ID/uni_Delivery_person_ID')
    return fig

#--------------------------------------------------------------------------------

def country_maps(df):
    df_aux6 = df.loc[: , ['Delivery_location_latitude','Delivery_location_longitude', 'City','Road_traffic_density']].groupby(['City', 'Road_traffic_density']).median().reset_index()
    #grafico
    map = folium.Map()
    for index, location_info in df_aux6.iterrows():
        folium.Marker( [location_info['Delivery_location_latitude'],
                        location_info['Delivery_location_longitude']],
                        popup=location_info[['City', 'Road_traffic_density']] ).add_to(map) 
    #mostrar
    folium_static(map, width=1024, height=600)

########################################################################################
#INICIO ESTRUTURA DO CODIGO

#import dataset
df1 = pd.read_csv('dataset/train.csv')
#df = df1.copy()
df = clean_code(df1)

pd.options.mode.copy_on_write = True

##########################################################################################
#Barra Lateral no streamlit
################################################################
image_path ='curry.png'
image = Image.open(image_path)
st.sidebar.image(image, width=120)


st.sidebar.markdown("# **Curry Company**")
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
#layout do streamlit
##########################################################################################
st.title('Marketplace - Visão Empresa')

tab1, tab2, tab3 = st.tabs(['Visão Estratégica', 'Visão Tática', 'Visão Geografica'])

with tab1:
    st.container()
        #VISÃO EMPRESA
    st.header('Pedidos por Dia')
    fig = order_day(df)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.container()
        st.markdown('### Pedidos por Intensidade de Tráfego')
        fig = traffic_order_share(df)
        st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.container()
        st.markdown('### Intensidade de Tráfego por Região')
        fig = traffic_order_city(df)
        st.plotly_chart(fig, use_container_width=True)
            
with tab2:
    with st.container():
        st.header('Pedidos por Semana')
        fig = order_week(df)
        st.plotly_chart(fig, use_container_width=True)

    
    with st.container():
        st.header('Pedidos por Entregador por Ssemana')
        fig = order_share_week(df)
        st.plotly_chart(fig, use_container_width=True)   
        
with tab3:
    st.header('Mapa')
    country_maps(df)








