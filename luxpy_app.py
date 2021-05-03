# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import base64
from io import BytesIO
from PIL import Image

# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

import luxpy as lx
from luxpy.toolboxes import photbiochem as ph

logo = plt.imread('LUXPY_logo_new1_small.png')

def spd_to_tm30(spd):
    return lx.cri._tm30_process_spd(spd, cri_type = 'ies-tm30')

def plot_tm30(data,
              source = '',
              manufacturer = '',
              date = '',
              model = '',
              notes = '',
              save_fig_name = None):
    return lx.cri.plot_tm30_report(data,
                            cri_type = 'ies-tm30',
                            source = source,
                            manufacturer = manufacturer,
                            data = date,
                            model = model,
                            notes = notes,
                            save_fig_name = save_fig_name)


def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/png;base64,{img_str}">Download result</a>'
	return href


def load_spectral_data():
    st.sidebar.markdown("Load spectral data:")
    st.sidebar.checkbox("Column format", True, key = 'options')
    units = st.sidebar.selectbox('Units',['W/nm [,.m²,.m³.sr, ...]','mW/nm [,.m²,.m³.sr, ...]' ])
    unit_factor = 1.0 if units == 'W/nm [,.m²,.m³.sr, ...]' else 1/1000
    header = 'infer' if st.sidebar.checkbox("Data file has header", False, key = 'header') else None
    sep = st.sidebar.selectbox('Separator',[',','\t',';'])

    uploaded_file = st.sidebar.file_uploader("Upload spectral data file",accept_multiple_files=False,type=['csv','dat','txt'])
    file_details = ''
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        df = pd.read_csv(uploaded_file, header =  header, sep = sep)
        df.iloc[:,df.columns[1]:] = df.iloc[:,df.columns[1]:] * unit_factor
        if header == 'infer':
            names = df.columns[1:]
        else:
            names = ['S{:1.0f}'.format(i) for i in range(len(df.columns)-1)]
        df.columns = ['nm'] + names
    else:
        df = pd.DataFrame(lx._CIE_D65.copy().T) # D65 default
        names = 'D65 (default spectrum)'
        df.columns = ['nm','D65 (default spectrum)']
        
        
    display = st.sidebar.selectbox("Display input data", ('Graph','DataFrame','No'))
    return df, file_details, display
    
def display_spectral_input_data(df, file_details, display):
    if display != 'No':
        st.sidebar.title('Spectral input data')
        st.sidebar.write(file_details)
    if display == 'No':
        pass
    elif display == 'DataFrame':
        st.sidebar.dataframe(df)
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        plt.sca(ax)
        lx.SPD(df.values.T).plot(wavelength_bar=False, label = df.columns[1:])
        ax.legend()
        st.sidebar.pyplot(fig)

def calculate(option, df):
    if option == 'ANSI/IESTM30':
        name = st.sidebar.selectbox('Select spectrum',df.columns[1:])
        index = list(df.columns[1:]).index(name)
        manufacturer = st.sidebar.text_input('Manufacturer','')
        date = st.sidebar.text_input('Date','')
        model = st.sidebar.text_input('Model','')
        notes = st.sidebar.text_input('Notes','')
        data = df.values.T[[0,index+1],:]
    elif option == 'Alpha-opic quantities (CIE S026)':
        data = df.values.T
    else:
        data = df.values.T
    
    if st.sidebar.button('Calculate ' + option):
        
        if option == 'ANSI/IESTM30':
            #data = df.values.T[[0,index+1],:] # spd_to_tm30(df.values.T[[0,index+1],:])
            axs, results = lx.cri.plot_tm30_report(data, 
                                                source = name, 
                                                manufacturer = manufacturer,
                                                date = date,
                                                model = model,
                                                notes = notes,
                                                save_fig_name = name)
            st.pyplot(axs['fig'])
            
            # img = plt.imread(name + '.png')
            # result = Image.fromarray((img[...,:-1]*255).astype(np.uint8))
            # st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        elif option == 'Alpha-opic quantities (CIE S026)':
            try:
                # alpha-opic Ee, -EDI, -DER and -ELR:
                cieobs = '1931_2'
                aEe = ph.spd_to_aopicE(data,cieobs = cieobs,actionspectra='CIE-S026',out = 'Eeas')
     
                aedi = ph.spd_to_aopicEDI(data,cieobs = cieobs,actionspectra='CIE-S026')
                ader = ph.spd_to_aopicDER(data, cieobs = cieobs,actionspectra='CIE-S026')
                aelr = ph.spd_to_aopicELR(data, cieobs = cieobs,actionspectra='CIE-S026')
                results = {'a-EDI':aedi,'aDER':ader,'aELR':aelr}
                
                quants = ['a-Ee, W/m²','a-EDI, lx','a-DER, a.u.','a-ELR, lm/W']
                tmp_q = np.repeat(quants,len(df.columns[1:]))
                tmp_n = np.tile(df.columns[1:],len(quants))
                df_indices = ['{:s}_({:s})'.format(name,quant) for name,quant in zip(tmp_n,tmp_q)]
                df_res = pd.DataFrame(np.vstack((aEe,aedi,ader,aelr)), 
                                      columns = ph._PHOTORECEPTORS,
                                      index = df_indices)
                st.dataframe(df_res)
            except:
                st.markdown('Not implemented yet (03/05/2021)')
        elif option == "(X,Y,Z), (x,y), (u',v'), (CCT,Duv)":
            
            pass
    else:
        data = []
    return data
            
            
def main():
    st.sidebar.image(logo, width=300)
    st.sidebar.markdown('## **Online calculator for lighting and color science**')
    
    st.sidebar.title('Control panel')
    option = st.sidebar.selectbox("Calculation options", ('ANSI/IESTM30','Alpha-opic quantities (CIE S026)'))
    
    if option in ('ANSI/IESTM30','Alpha-opic quantities (CIE S026)'):
        df, file_details, display = load_spectral_data()
        display_spectral_input_data(df, file_details, display)
        calculate(option, df)
    else:
        df, file_details, display = None, {}, 'No'
    
    
if __name__ == '__main__':
    main()
    
    