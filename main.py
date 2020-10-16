import io

import numpy as np
import streamlit as st
from PIL import Image

from detedge import lpf, hpf, bpf, detect_edges
from detedgewev import detect_edges_wavelet as detect_edges_wav

if __name__ == '__main__':

    option = st.sidebar.selectbox('Options', options=['-', 'Fourier', 'Wavelet'], index=0)

    if option == '-':
        st.markdown("""
                # Proyecto de Matematica Numerica
                ## Tema:
                    Deteccion de borde y eliminacion de ruidos en imagenes usando FFT
                ### Autores:
                    Alejandro Klever Clemente
                    Miguel Angel Gonzalez Calles
                    Laura Tamayo Blanco
            """)
    elif option == 'Fourier':
        file: io.BytesIO = st.sidebar.file_uploader('Upload Photo', type=["jpg", "jpeg", "png"])
        if file is not None:
            img: np.ndarray = np.array(Image.open(file).convert('L'))
            st.image(image=img, caption='Input Image', use_column_width=True)

            option = st.sidebar.selectbox('Select Mask',
                                          options=[
                                              'low pass filter',
                                              'high pass filter',
                                              'band pass filter'],
                                          index=0
                                          )

            if 'low pass filter' == option:
                lpf_radio = st.sidebar.slider('low frequency radio', max_value=100., value=9., step=.01)
                img_back, (spectrum, mask) = detect_edges(img, lpf, lpf_radio * min(img.shape) / 100)
            elif 'high pass filter' == option:
                hpf_radio = st.sidebar.slider('high frequency radio', max_value=100., value=9., step=.01)
                img_back, (spectrum, mask) = detect_edges(img, hpf, hpf_radio * min(img.shape) / 100)
            else:
                rin, rout = st.sidebar.slider('band frequency radio', max_value=100., value=(5., 12.), step=.01)
                img_back, (spectrum, mask) = detect_edges(img, bpf,
                                                          (rin * min(img.shape) / 100, rout * min(img.shape) / 100)
                                                          )

            st.sidebar.image(image=np.uint8(spectrum), caption='After FFT', use_column_width=True)
            st.sidebar.image(image=np.uint8(mask), caption='FFT + MASK', use_column_width=True)
            st.image(image=np.uint8(img_back), caption='After FFT Inverse', use_column_width=True)
    else:
        file: io.BytesIO = st.sidebar.file_uploader('Upload Photo', type=["jpg", "jpeg", "png"])
        img: np.ndarray = np.array(Image.open(file).convert('L'))
        st.image(image=img, caption='Input Image', use_column_width=True)

        option = st.sidebar.selectbox('Select Mask',
                                      options=[
                                          'low pass filter',
                                          'high pass filter',
                                          'band pass filter'],
                                      index=0
                                      )
        if 'low pass filter' == option:
            lpf_radio = st.sidebar.slider('low frequency radio', max_value=100., value=40., step=.01)
            img_back, (spectrum, mask) = detect_edges_wav(img, lpf, lpf_radio * min(img.shape) / 100)
        elif 'high pass filter' == option:
            hpf_radio = st.sidebar.slider('high frequency radio', max_value=100., value=30., step=.01)
            img_back, (spectrum, mask) = detect_edges_wav(img, hpf, hpf_radio * min(img.shape) / 100)
        else:
            rin, rout = st.sidebar.slider('band frequency radio', max_value=100., value=(10., 20.), step=.01)
            img_back, (spectrum, mask) = detect_edges_wav(img, bpf, (rin * min(img.shape) / 100, rout * min(img.shape) / 100))
        st.sidebar.image(image=np.uint8(spectrum), caption='After Analysis ', use_column_width=True)
        st.sidebar.image(image=np.uint8(mask), caption='Coefficients + MASK', use_column_width=True)
        st.image(image=np.uint8(img_back), caption='After Synthesis', use_column_width=True)
