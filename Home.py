import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="icon"
)

######################################################################################
#Barra Lateral no streamlit
################################################################
image_path ='curry.png'
image = Image.open(image_path)
st.sidebar.image(image, width=120)

st.sidebar.markdown("# **Curry Company**")
st.sidebar.markdown("### _O Delivery mais rápidico da Região_")
st.sidebar.markdown("""---""")
st.sidebar.markdown('Powerd by HLF.Meigabot')

##########################################################################################
#layout do streamlit
##########################################################################################
st.write("# Curry Companie Growth Dashboard")

st.markdown(
    """
    Growth Dashboard foi construido para acompanhar as métricas e indicadores do negócio:
    - Visão Empresa:
        - Visão Gerencial: Métricas gerais
        - Visão tática: inidicadores de crescimento semanais
        - Visão geográfica: insights de localização
    - Visão Entregadores:
        - Inidicadores de crescimento semanais
    - Visão Restaurantes:
        - Inidicadores de movimentação semanais
    ### Helpdesk:
    - rayssarpj@gmail.com
    """
    
)