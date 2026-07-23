import streamlit as st
from PIL import Image
from pathlib import Path
import os
import style

try:
    from fast_neural_style.neural_style import style
except ImportError:
    import style
    
# Configuración de la página
st.set_page_config(
    page_title="Transferencia de Estilo Neural",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS mejorado
st.markdown("""
<style>
    /* Estilo para los encabezados */
    h3 {
        margin-bottom: 0.4rem;
        padding-bottom: 0.4rem;
    }
    
    /* Navegación superior */
    .nav-container {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 999;
        background: rgba(38, 39, 48, 0.9);
        border-radius: 12px;
        padding: 8px 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        backdrop-filter: blur(4px);
        border: 1px solid #3a3b45;
    }
    .nav-item {
        display: inline-block;
        margin: 0 6px;
        padding: 6px 12px;
        color: #a1a5b0;
        font-weight: 500;
        font-size: 0.9rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-item:hover {
        color: #ffffff;
        background: rgba(74, 75, 92, 0.5);
    }
    .nav-item.active {
        color: #ffffff;
        background: #4b4d5c;
    }
    .nav-separator {
        display: inline-block;
        color: #3a3b45;
        margin: 0 2px;
    }
    
    /* Botón principal */
    div.stButton > button:first-child {
        display: block;
        width: 100%;
        margin-top: 2em;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Ajustes para las imágenes */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
            
    .result-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Navegación superior
current_page = Path(st.query_params.get("page", ["main"])[0]).stem
nav_html = f"""
<div class="nav-container">
    <div class="nav-item {'active' if current_page == 'main' else ''}" 
         onclick="window.location.pathname='/'">Main</div>
    <span class="nav-separator">|</span>
    <div class="nav-item {'active' if current_page == 'Info' else ''}" 
         onclick="window.location.pathname='/Info'">Info</div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# Título principal
st.markdown("<h1 style='text-align: center; margin-bottom: 1.5rem;'>🎨 Transferencia de Estilo de Imagen</h1>", 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Configuración")
    upload_option = st.radio(
        "Seleccionar imagen de contenido",
        ("Usar imagen de ejemplo", "Subir mi propia imagen")
    )
    
    if upload_option == "Subir mi propia imagen":
        uploaded_file = st.file_uploader(
            "Sube tu imagen (JPEG/PNG)", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    else:
        uploaded_file = None
    
    style_name = st.selectbox(
        'Seleccionar estilo artístico',
        ('candy', 'mosaic', 'rain_princess', 'udnie','monet','michelangelo')
    )

# Layout principal
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align: center; margin-bottom: 0.8em;'>Imagen de Contenido</h3>", 
               unsafe_allow_html=True)
    
    if upload_option == "Subir mi propia imagen" and uploaded_file is not None:
        content_img = Image.open(uploaded_file)
        st.image(content_img, width=400)
        input_image = uploaded_file
    else:
        try:
            default_img = "content.jpg"
            content_img = Image.open(f"images/content-images/{default_img}")
            st.image(content_img, width=400)
            input_image = f"images/content-images/{default_img}"
        except FileNotFoundError:
            st.error("Imagen de ejemplo no encontrada")

with col2:
    st.markdown("<h3 style='text-align: center; margin-bottom: 0.8em;'>Estilo de Referencia</h3>", 
               unsafe_allow_html=True)
    
    try:
        style_img = Image.open(f"images/style-images/{style_name}.jpg")
        st.image(style_img, width=367)
    except FileNotFoundError:
        st.error(f"Imagen de estilo no encontrada: {style_name}")

# Botón de transformación
clicked = st.button("✨ Aplicar Transformación Estilística", type="primary")

# Procesamiento
if clicked:
    try:
        with st.spinner(f"Aplicando estilo {style_name}..."):
            model = style.load_model(f"saved_models/{style_name}.pth")
            
            if isinstance(input_image, str):
                output_image = f"images/output-images/{style_name}-{os.path.basename(input_image)}"
                style.stylize(model, input_image, output_image)
            else:
                output_image = f"images/output-images/{style_name}-{uploaded_file.name}"
                with open(output_image, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                style.stylize(model, output_image, output_image)
        
        st.success("¡Transformación completada!")
        
        # Resultado estilizado
        st.markdown("<h3 style='text-align: center; margin-top: 2rem;'>Resultado Estilizado</h3>", 
                   unsafe_allow_html=True)
        
        try:
            result_img = Image.open(output_image)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(result_img, width=500)
                with open(output_image, "rb") as file:
                    st.download_button(
                        label="⬇️ Descargar Imagen",
                        data=file,
                        file_name=f"estilo_{style_name}_{os.path.basename(output_image)}",
                        mime="image/png",
                        use_container_width=True
                    )
        except FileNotFoundError:
            st.error("No se pudo generar la imagen de salida")
            
    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")