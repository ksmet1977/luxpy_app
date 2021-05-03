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
    units = st.sidebar.selectbox('Units',['W/nm [,.m²,.m².sr, ...]','mW/nm [,.m²,.m².sr, ...]' ])
    unit_factor = 1.0 if units == 'W/nm [,.m²,.m².sr, ...]' else 1/1000
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
        names = 'D65 (default)'
        df.columns = ['nm',names]
        
        
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
    global start
    if option == 'ANSI/IESTM30 graphic report':
        name = st.sidebar.selectbox('Select spectrum',df.columns[1:])
        index = list(df.columns[1:]).index(name)
        manufacturer = st.sidebar.text_input('Manufacturer','')
        date = st.sidebar.text_input('Date','')
        model = st.sidebar.text_input('Model','')
        notes = st.sidebar.text_input('Notes','')
        data = df.values.T[[0,index+1],:]
    elif ((option == 'Alpha-opic quantities (CIE S026)') | 
         (option == 'ANSI/IESTM30 quantities') |
         (option == 'CIE 13.3-1995 Ra, Ri quantities') |
         (option == 'CIE 224:2017 Rf, Rfi quantities')):
        spd_opts = ['all'] + list(df.columns[1:])
        name = st.sidebar.selectbox('Select spectrum',spd_opts)
        if name != 'all':
            index = spd_opts.index(name) - 1
            data = df.values.T[[0,index+1],:]
            names = [df.columns[1:][index]]
        else:
            data = df.values.T
            names = df.columns[1:]
    else:
        data = df.values.T
    
        
    if st.sidebar.button('Calculate ' + option):
        start = False
        if option == 'ANSI/IESTM30 graphic report':
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
        elif option == 'ANSI/IESTM30 quantities':
            d = spd_to_tm30(data)
            xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
            uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
            quants = ['CCT','Duv'] + ['x','y',"u'","v'"] + ['Rf', 'Rg']
            quants += ['Rcsh{:1.0f}'.format(i+1) for i in range(d['Rcshj'].shape[0])] 
            quants += ['Rhsh{:1.0f}'.format(i+1) for i in range(d['Rhshj'].shape[0])] 
            quants += ['Rfh{:1.0f}'.format(i+1) for i in range(d['Rfhj'].shape[0])]
            quants += ['Rf{:1.0f}'.format(i+1) for i in range(d['Rfi'].shape[0])]
            
            df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                             xy.T,uv.T,
                                             d['Rf'], d['Rg'],
                                             d['Rcshj'],d['Rhshj'],d['Rfhj'],d['Rfi'])).T,
                                   columns = quants,
                                   index = names)
            
            st.markdown('**ANSI/IES TM30 Quantities (CCT, Duv, Rf,Rg, ...)**')
            st.dataframe(df_res)
            
            st.markdown('*CCT: Correlated Color Temperature (K)*')
            st.markdown('*Duv: distance from Planckian locus*')
            st.markdown('*xy: CIE 1931 2° xy chromaticity coordinates of illuminant white point*')
            st.markdown("*u'v': CIE 1976 2° u'v' chromaticity coordinates*")
            st.markdown('*Rf: general color fidelity index*')
            st.markdown('*Rg: gamut area index*')
            st.markdown('*Rcshj: local chroma shift for hue bin j*')
            st.markdown('*Rhshj: local hue shift for hue bin j*')
            st.markdown('*Rfhj: local color fidelity index for hue bin j*')
            st.markdown('*Rfi: sample color fidelity index for sample i*')
        
        elif option == 'CIE 13.3-1995 Ra, Ri quantities':
            d = spd_to_tm30(data)
            Ra, _ = lx.cri.spd_to_cri(d['St'], cri_type = 'ciera', out = 'Rf,Rfi')
            _, Ri = lx.cri.spd_to_cri(d['St'], cri_type = 'ciera-14', out = 'Rf,Rfi')

            xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
            uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
            quants = ['CCT','Duv'] + ['x','y',"u'","v'"] + ['Ra']
            quants += ['R{:1.0f}'.format(i+1) for i in range(Ri.shape[0])]
            
            df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                             xy.T,uv.T,
                                             Ra, Ri)).T,
                                   columns = quants,
                                   index = names)
            
            st.markdown('**CIE 13.3-1995 Ra, Ri quantities**')
            st.dataframe(df_res)
            
            st.markdown('*CCT: Correlated Color Temperature (K)*')
            st.markdown('*Duv: distance from Planckian locus*')
            st.markdown('*xy: CIE 1931 2° xy chromaticity coordinates of illuminant white point*')
            st.markdown("*u'v': CIE 1976 2° u'v' chromaticity coordinates*")
            st.markdown('*Ra: general color fidelity index*')
            st.markdown('*Ri: specific color fidelity index for sample i*')
  
        elif option == 'CIE 224:2017 Rf, Rfi quantities':
            d = spd_to_tm30(data)
            Ra,Ri = lx.cri.spd_to_cierf(d['St'], out = 'Rf,Rfi')
            xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
            uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
            quants = ['CCT','Duv'] + ['x','y',"u'","v'"] + ['Rf']
            quants += ['Rf{:1.0f}'.format(i+1) for i in range(Ri.shape[0])]
            
            df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                             xy.T,uv.T,
                                             Ra, Ri)).T,
                                   columns = quants,
                                   index = names)
            
            st.markdown('**CIE 224:2017 Rf, Rfi quantities**')
            st.dataframe(df_res)
            
            st.markdown('*CCT: Correlated Color Temperature (K)*')
            st.markdown('*Duv: distance from Planckian locus*')
            st.markdown('*xy: CIE 1931 2° xy chromaticity coordinates of illuminant white point*')
            st.markdown("*u'v': CIE 1976 2° u'v' chromaticity coordinates*")
            st.markdown('*Rf: general color fidelity index*')
            st.markdown('*Rfi: specific color fidelity index for sample i*')

    
        elif option == 'Alpha-opic quantities (CIE S026)':
            try:
                # alpha-opic Ee, -EDI, -DER and -ELR:
                cieobs = '1931_2'
                aEe = ph.spd_to_aopicE(data,cieobs = cieobs,actionspectra='CIE-S026',out = 'Eeas')
     
                aedi = ph.spd_to_aopicEDI(data,cieobs = cieobs,actionspectra='CIE-S026')
                ader = ph.spd_to_aopicDER(data, cieobs = cieobs,actionspectra='CIE-S026')
                aelr = ph.spd_to_aopicELR(data, cieobs = cieobs,actionspectra='CIE-S026')
                results = {'a-EDI':aedi,'aDER':ader,'aELR':aelr}
                
                quants = ['a-Ee','a-EDI','a-DER','a-ELR']
                tmp_q = np.repeat(quants,len(names))
                tmp_n = np.tile(names,len(quants))
                df_indices = ['{:s}: {:s}'.format(quant,name) for name,quant in zip(tmp_n,tmp_q)]
                df_res = pd.DataFrame(np.vstack((aEe,aedi,ader,aelr)), 
                                      columns = ph._PHOTORECEPTORS,
                                      index = df_indices)
                
                st.markdown('**alpha-opic quantities (CIE S026)**')
                st.dataframe(df_res)
                
                st.markdown('*Ee: irradiance (W/m²)*')
                st.markdown('*EDI: Equivalent Daylight Illuminance (lux)*')
                st.markdown('*DER: Daylight Efficacy Ratio*')
                st.markdown('*ELR: Efficacy of Luminous Radiation (W/lm)*')
                
                
            except:
                st.markdown('Not implemented yet (03/05/2021)')
        elif option == "(X,Y,Z), (x,y), (u',v'), (CCT,Duv)":
            
            pass

    else:
        start = False
        data = []
    return data, start
 

start = True           
            
def main():
    global start 
    st.sidebar.image(logo, width=300)
    link = '[github.com/ksmet1977/luxpy](http://github.com/ksmet1977/luxpy)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    st.sidebar.markdown('## **Online calculator for lighting and color science**')
    st.sidebar.markdown("""---""")
    st.sidebar.title('Control panel')
    option = st.sidebar.selectbox("Calculation options", ('',
                                                          'ANSI/IESTM30 quantities', 
                                                          'ANSI/IESTM30 graphic report',
                                                          'CIE 13.3-1995 Ra, Ri quantities',
                                                          'CIE 224:2017 Rf, Rfi quantities',
                                                          'Alpha-opic quantities (CIE S026)'))
    
    if option in ('ANSI/IESTM30 quantities',
                  'ANSI/IESTM30 graphic report',
                  'CIE 13.3-1995 Ra, Ri quantities',
                  'CIE 224:2017 Rf, Rfi quantities',
                  'Alpha-opic quantities (CIE S026)'):
        df, file_details, display = load_spectral_data()
        display_spectral_input_data(df, file_details, display)
        data,start = calculate(option, df)
    else:

        df, file_details, display = None, {}, 'No'
        
    if start:
        st.markdown('### Usage:')
        st.markdown(' 1. Select calculation option.')
        st.markdown(' 2. Load data + set data details.')
        st.markdown(' 3. Press calculate.')
        
    st.markdown("""---""")
    st.markdown("If you use **LUXPY**, please cite the following tutorial paper published in LEUKOS:")
    st.markdown("**Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23.** DOI: [10.1080/15502724.2018.1518717](10.1080/15502724.2018.1518717)")
    start = False
    
if __name__ == '__main__':
    main()
    
    