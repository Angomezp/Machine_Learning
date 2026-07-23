import streamlit as st
from PIL import Image
import os

# Configuración de página
st.set_page_config(
    page_title="Galería de Estilos Artísticos",
    layout="wide",
    page_icon="🎨"
)

# CSS mejorado
st.markdown("""
<style>
    .main-header {
        background: rgba(38, 39, 48, 0.9);
        color: white;
        padding: 2rem;
        border-radius: 0;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 4px solid #f63366;
    }
    .style-title {
        color: #f63366;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-align: center;
        width: 100%;
    }
    .style-subtitle {
        color: #a1a5b0;
        font-size: 1.1rem;
        margin-bottom: 0;
        text-align: center;
    }
    .style-img-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .meta-info {
        color: #a1a5b0;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .description-text {
        text-align: justify;
        margin: 1rem 0;
        padding: 0 1rem;
        color: #f0f2f6;
    }
    .column-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    .style-grid {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="style-title">🎨 Catálogo de Estilos</h1>
    <p class="style-subtitle">Transforma tus imágenes con transferencia de estilo neuronal</p>
</div>
""", unsafe_allow_html=True)


ESTILOS = {
        " El paseo ": {
        "imagen": "images/style-images/monet.jpg",
        "descripcion": " El Paseo (también conocido como La Mujer con Sombrilla o Madame Monet y su Hijo) es una de las obras más icónicas de Claude Monet. Representa a su primera esposa, Camille Monet, "
        "y a su hijo Jean paseando por un campo en Argenteuil, cerca de París. Camille aparece de pie en una colina, vestida con un elegante traje blanco y sosteniendo una sombrilla que proyecta una sombra "
        "sobre su rostro. La escena captura un momento efímero, con el viento moviendo su vestido y la hierba del campo, transmitiendo una sensación de movimiento y vida.",
        "tecnica": "	Óleo sobre tela con pinceladas rápidas y sueltas, colores vibrantes, contraste de luz/sombra todas caracteristicas tipicas del estilo impresionista.",
        "año": "1875"
    },
        "Udnie, Young American Girl": {
        "imagen": "images/style-images/udnie.jpg",
        "descripcion": "Se cree que esta pintura se inspiró en la interpretación de una danza de estilo hindú de la bailarina polaca Stasia Napierkowska"
        "Se le considera una obra maestra de la abstracción temprana que fusiona cubismo, orfismo y futurismo en una explosión de formas dinámicas. ",
        "tecnica": "Pintura al óleo sobre lienzo",
        "año": "1913"
    },

    "Candy": {
        "imagen": "images/style-images/candy.jpg",
        "descripcion": " Transforma imágenes en composiciones ultra-coloridas y oníricas, con colores saturados que parecen derretirse como caramelos. Los bordes se difuminan creando un efecto dulce y surrealista, ideal para retratos fantásticos o paisajes de ensueño ",
        "tecnica": "Desconocida",
        "año": "2016"
    },

        "Rain Princess": {
        "imagen": "images/style-images/rain_princess.jpg",
        "descripcion": "Obra mas famosa del pintor bielorruso Leonid Afremov, la cual tiene un efecto melancólico y atmosférico que recuerda a pinturas impresionistas de días lluviosos, con tonos"
        " azulados y pinceladas fluidas.  El autor se caracteriza por nunca hacer uso de pincel, siendo esta una obra creada unicamente con espatula. ",
        "tecnica": "Oleo sobre lienzo la cual  evoca la melancolía y vitalidad de las pinturas impresionistas ",
        "año": "2003"
    },
    
    "Mosaic": {
        "imagen": "images/style-images/mosaic.jpg",
        "descripcion": "Transforma imágenes en composiciones que imitan los mosaicos clásicos, fragmentando la realidad en pequeñas teselas digitales de colores vibrantes. Cada detalle se reconstruye mediante patrones geométricos, creando un efecto entre lo artesanal y lo pixelado, como una versión moderna de los antiguos mosaicos romanos o bizantinos.",
        "tecnica": "Desconocido",
        "año": "Inspiración siglo VI"
    },


    "La creación de Adán": {
        "imagen": "images/style-images/michelangelo.jpg",
        "descripcion": "Fresco icónico de la Capilla Sixtina que representa el momento en que Dios da vida a Adán. Las figuras divinas y humanas se conectan a través de un gesto casi tangible, con composición dramática y fuerza visual. El contraste entre la energía celestial y la languidez terrenal de Adán crea una narrativa poderosa.",
        "tecnica": "Pintura al fresco con dominio anatómico extremo, colores vibrantes (revelados tras restauración) y dinamismo en las figuras.",
        "año": "1511"
    }

}

# Grid de estilos
st.markdown('<div class="style-grid">', unsafe_allow_html=True)

for i, (estilo, info) in enumerate(ESTILOS.items()):
    # Alternar la posición de la imagen
    if i % 2 == 0:
        col1, col2 = st.columns([1, 2])
    else:
        col2, col1 = st.columns([2, 1])
    
    with st.container():
        with col1:
            # Contenedor para centrar la imagen verticalmente
            st.markdown('<div class="style-img-container">', unsafe_allow_html=True)
            try:
                if os.path.exists(info["imagen"]):
                    img = Image.open(info["imagen"])
                    st.image(img, use_container_width=True)
                else:
                    st.warning(f"Imagen no encontrada: {info['imagen']}")
            except Exception as e:
                st.error(f"Error al cargar imagen: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Contenedor para el texto centrado verticalmente
            st.markdown('<div class="column-container">', unsafe_allow_html=True)
            st.markdown(f'<h3 class="style-title" style="text-align: center;">🎨 {estilo}</h3>', unsafe_allow_html=True)
            st.markdown(f'<p class="meta-info"><strong>📅 Año:</strong> {info["año"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="meta-info"><strong>🛠️ Técnica:</strong> {info["tecnica"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="description-text">{info["descripcion"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.markdown("### ¿Listo para transformar tus imágenes?")
if st.button("Ir a la Aplicación Principal", type="primary"):
    st.switch_page("main.py")