# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import base64
from io import BytesIO, StringIO
from PIL import Image


import luxpy as lx
from luxpy.toolboxes import photbiochem as ph
from luxpy.toolboxes import iolidfiles as lid 

logo = plt.imread('LUXPY_logo_new1_small.png')

__version__ = 'v0.0.27'

def get_table_download_link_csv(df):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="luxpy_app_download.csv" target="_blank">Download csv file</a>'
    return href

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
# 	href = f'<a href="data:file/jpg;base64,{img_str}">luxpy_app_download.csv</a>'
	href = f'<a href="data:file/jpg;base64,{img_str}" download="luxpy_app_download.png" target="_blank">Download image file</a>'
	return href

#------------------------------------------------------------------------------
# Sidebar sections
#------------------------------------------------------------------------------

# Data loaders:
#--------------

def load_spectral_data():
    # Set title for this sidebar section:
    st.sidebar.markdown("""---""")
    st.sidebar.markdown("### Load spectral data:")
    
    # expander with data format options:
    expdr_dopts = st.sidebar.beta_expander("Data-format options")
    expdr_dopts.checkbox("Column format", True, key = 'options')
    units = expdr_dopts.selectbox('Units',['W/nm [,.m²,.m².sr, ...]','mW/nm [,.m²,.m².sr, ...]' ])
    unit_factor = 1.0 if units == 'W/nm [,.m²,.m².sr, ...]' else 1/1000
    header = 'infer' if expdr_dopts.checkbox("Data file has header", False, key = 'header') else None
    sep = expdr_dopts.selectbox('Separator',[',','\t',';'])
    
    # expander with data loading:
    expdr_dload = st.sidebar.beta_expander("Upload Spectral Data",True)
    uploaded_file = expdr_dload.file_uploader("",accept_multiple_files=False,type=['csv','dat','txt'])
    file_details = ''
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        df = pd.read_csv(uploaded_file, header =  header, sep = sep) # read in data
        df.iloc[:,df.columns[1]:] = df.iloc[:,df.columns[1]:] * unit_factor # correct data for units
        names = df.columns[1:] if (header == 'infer') else ['S{:1.0f}'.format(i+1) for i in range(len(df.columns)-1)]
        df.columns = ['nm'] + names
    else:
        df = pd.DataFrame(lx._CIE_D65.copy().T) # D65 default
        names = 'D65 (default)'
        df.columns = ['nm',names] 
    return df, file_details

def load_LID_data():
    # Set title for this sidebar section:
    st.sidebar.markdown("""---""")
    st.sidebar.markdown("### Load LID data:")
    
    expdr_dload = st.sidebar.beta_expander("Upload LID (IES/LDT) data file",True)
    uploaded_file = expdr_dload.file_uploader("",accept_multiple_files=False,type=['ies','ldt'])
    file_details = {"FileName":'',"FileType":'',"FileSize":''}
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        LID = lid.read_lamp_data(stringio.read(), verbosity = 1) # read in data
        expdr_dload.write(file_details)
    else:
        LID = None
        st.sidebar.text('No LID data file selected.')
        st.sidebar.text('Load file first!')
    return LID, file_details

# loaded data displayers:
#------------------------

placeholder_spdselector = None
def display_spectral_input_data(df, file_details, sidebar = True):
    st.sidebar.markdown('### Input data:')
    expdr_dshow = st.sidebar.beta_expander('Show input data') if sidebar  else st.beta_expander('Show input data')      
    display = expdr_dshow.selectbox("Display format", ('Graph','DataFrame'))

    expdr_dshow.write(file_details)
    if display == 'DataFrame':
        expdr_dshow.dataframe(df)
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        plt.sca(ax)
        lx.SPD(df.values.T).plot(wavelength_bar=False, label = df.columns[1:])
        ax.legend()
        expdr_dshow.pyplot(fig)
        
    # global placeholder_spdselector
    # placeholder_spdselector = st.sidebar.empty() # placeholder for spetrum name selector
    

def generate_LID_plots(LID):
    
        # or combine draw and render (but use only 2D image):
    if LID is not None:
        # generate LID figure:
        fig = plt.figure(figsize=[14,7])
        axs = [fig.add_subplot(121, projection = 'polar'),
               fig.add_subplot(122)]
        lid.draw_lid(LID, ax = axs[0])
        Lv2D = lid.render_lid(LID, sensor_resolution = 100,
                        sensor_position = [0,-1,0.8], sensor_n = [0,1,-0.2], fov = (90,90), Fd = 2,
                        luminaire_position = [0,1.3,2], luminaire_n = [0,0,-1],
                        wall_center = [0,2,1], wall_n = [0,-1,0], wall_width = 4, wall_height = 2, wall_rho = 1,
                        floor_center = [0,1,0], floor_n = [0,0,1], floor_width = 4, floor_height = 2, floor_rho = 1,
                        ax3D = None, ax2D = axs[1], join_axes = False, 
                        plot_luminaire_position = True, plot_lumiaire_rays = False, plot_luminaire_lid = True,
                        plot_sensor_position = True, plot_sensor_pixels = False, plot_sensor_rays = False, 
                        plot_wall_edges = True, plot_wall_luminance = True, plot_wall_intersections = False,
                        plot_floor_edges = True, plot_floor_luminance = True, plot_floor_intersections = False,
                        out = 'Lv2D')
    else:
        fig = None
    
    return fig

#------------------------------------------------------------------------------
# Run classes and functions
#------------------------------------------------------------------------------
def spd_to_tm30(spd):
    return lx.cri._tm30_process_spd(spd, cri_type = 'ies-tm30')

def calc_tm30_quants(data, names, **kwargs):
    d = spd_to_tm30(data)
    LER = lx.spd_to_ler(d['St'])
    xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
    uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
    quants = ['CCT','Duv'] + ['x','y',"u'","v'",'LER'] + ['Rf', 'Rg']
    quants += ['Rcsh{:1.0f}'.format(i+1) for i in range(d['Rcshj'].shape[0])] 
    quants += ['Rhsh{:1.0f}'.format(i+1) for i in range(d['Rhshj'].shape[0])] 
    quants += ['Rfh{:1.0f}'.format(i+1) for i in range(d['Rfhj'].shape[0])]
    quants += ['Rf{:1.0f}'.format(i+1) for i in range(d['Rfi'].shape[0])]
    
    df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                     xy.T,uv.T,LER.T,
                                     d['Rf'], d['Rg'],
                                     d['Rcshj'],d['Rhshj'],d['Rfhj'],d['Rfi'])).T,
                           columns = quants,
                           index = names)
    legend = ['CCT','Duv','xy','uv','LER','Rf','Rg','Rcshj','Rhshj','Rfhj','Rfi']
    return df_res, legend, d

def plot_tm30_report(data, names, **kwargs):
    source = kwargs.get('source','')
    manufacturer = kwargs.get('manufacturer','')
    date = kwargs.get('date','')
    model = kwargs.get('model','')
    notes = kwargs.get('notes','')
    df_res, legend, d = calc_tm30_quants(data, names)

    axs, results = lx.cri.plot_tm30_report(d, 
                                            source = source, 
                                            manufacturer = manufacturer,
                                            date = date,
                                            model = model,
                                            notes = notes,
                                            save_fig_name = None)
    # legend = None
    return df_res, legend, axs
    
def calc_ciera_quants(data, names, **kwargs):
    d = spd_to_tm30(data)
    LER = lx.spd_to_ler(d['St'])
    Ra, _ = lx.cri.spd_to_cri(d['St'], cri_type = 'ciera', out = 'Rf,Rfi')
    _, Ri = lx.cri.spd_to_cri(d['St'], cri_type = 'ciera-14', out = 'Rf,Rfi')

    xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
    uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
    quants = ['CCT','Duv'] + ['x','y',"u'","v'",'LER'] + ['Ra']
    quants += ['R{:1.0f}'.format(i+1) for i in range(Ri.shape[0])]
    
    df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                     xy.T,uv.T,LER.T,
                                     Ra, Ri)).T,
                           columns = quants,
                           index = names)
    legend = ['CCT','Duv','xy','uv','LER','Ra','Ri']
    return df_res, legend, None

def calc_cierf_quants(data, names, **kwargs):
    d = spd_to_tm30(data)
    LER = lx.spd_to_ler(d['St'])
    Ra,Ri = lx.cri.spd_to_cierf(d['St'], out = 'Rf,Rfi')
    xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
    uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
    quants = ['CCT','Duv'] + ['x','y',"u'","v'",'LER'] + ['Rf']
    quants += ['Rf{:1.0f}'.format(i+1) for i in range(Ri.shape[0])]
    
    df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                     xy.T,uv.T, LER.T,
                                     Ra, Ri)).T,
                           columns = quants,
                           index = names)
    legend =  ['CCT','Duv','xy','uv','LER','Rf','Rfi']
    return df_res, legend, None
    
def calc_cies026_quants(data, names, **kwargs):
    # alpha-opic Ee, -EDI, -DER and -ELR:
    cieobs = kwargs.get('cieobs',lx._CIEOBS)
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
    legend = ['Ee', 'EDI', 'DER', 'ELR']
    return df_res, legend, None

def calc_colorimetric_quants(data, names, **kwargs):
    rfl = kwargs.get('rfl',None)
    xyz, xyzw = lx.spd_to_xyz(data, cieobs = kwargs['cieobs'], relative = kwargs['relative_xyz'], rfl = kwargs.get('rfl',None), out = 2)
    if rfl is not None: xyz = xyz[:,0,:] # get rid of light source dimension
    cct, duv = lx.xyz_to_cct(xyz, out ='cct,duv')
    xy = lx.xyz_to_Yxy(xyz)[...,1:]
    uv = lx.xyz_to_Yuv(xyz)[...,1:]
    quants = ['X','Y','Z'] + ['x','y',"u'","v'"] + ['CCT','Duv']
    
    df_res = pd.DataFrame(np.vstack((xyz.T,
                                     xy.T,uv.T,
                                     cct.T, duv.T,
                                     )).T,
                           columns = quants,
                           index = names)
    legend = ['XYZ','xy','uv','CCT','Duv']
    return df_res, legend, xyzw

def plot_ies_ldt_lid(LID, names, **kwargs):
    fig = generate_LID_plots(LID)
    return (None, None, fig)

# run options --> {option : (short name, function, input datatype, has_legend, title)}
run_options = {'' : ('', None, None, False, ''),
                'ANSI/IESTM30 graphic report' : ('tm30_report', plot_tm30_report, 'spd', False, 'ANSI/IESTM30 graphic report'),
               'ANSI/IESTM30 quantities' : ('tm30_quants' ,calc_tm30_quants, 'spd', True, '**ANSI/IES TM30 Quantities (CCT, Duv, Rf,Rg, ...)**'),
               'CIE 13.3-1995 Ra, Ri quantities' : ('ciera', calc_ciera_quants, 'spd',True,'**CIE 13.3-1995 Ra, Ri quantities**'),
               'CIE 224:2017 Rf, Rfi quantities': ('cierf', calc_cierf_quants, 'spd', True,'**CIE 224:2017 Rf, Rfi quantities**'),
               "SPD->(X,Y,Z), (x,y), (u',v'), (CCT,Duv)" :('colorimetric_quants',calc_colorimetric_quants, 'spd', True,"**Colorimetric quantities: (X,Y,Z), (x,y), (u',v'), (CCT,Duv)**"),
               'Alpha-opic quantities (CIE S026)' : ('cies2036_quants', calc_cies026_quants, 'spd', True, 'Alpha-opic quantities (CIE S026)'),
               'Plot Luminous Intensity Distribution (IES/LDT files)' : ('lid_plots', plot_ies_ldt_lid, 'lid', False, '**Luminous Intensity Distiribution (polar plot and render)**')
               }

legend_dict =  {'XYZ' : '*XYZ: CIE X,Y,Z tristimulus values*',
                'CCT' : '*CCT: Correlated Color Temperature (K)*',
                'Duv' : '*Duv: distance from Planckian locus*',
                'xy' :  '*xy: CIE 1931 2° xy chromaticity coordinates of illuminant white point*',
                'uv' :  "*u'v': CIE 1976 2° u'v' chromaticity coordinates*",
                'LER' : '*LER: Luminous Efficacy of Radiation (lm/W)*',
                'Rf' :  '*Rf: general color fidelity index*',
                'Rg' : '*Rg: gamut area index*',
                'Rcshj' : '*Rcshj: local chroma shift for hue bin j*',
                'Rhshj': '*Rhshj: local hue shift for hue bin j*',
                'Rfhj' :'*Rfhj: local color fidelity index for hue bin j*',
                'Rfi' : '*Rfi: specific color fidelity index for sample i*',
                'Ra' : '*Ra: general color fidelity index*',
                'Ri' : '*Ri: specific color fidelity index for sample i*',
                'Rf' : '*Rf: general color fidelity index*',
                'Rfi' : '*Rfi: specific color fidelity index for sample i*',
                'Ee' :  '*Ee: irradiance (W/m²)*',
                'EDI' : '*EDI: Equivalent Daylight Illuminance (lux)*',
                'DER' : '*DER: Daylight Efficacy Ratio*',
                'ELR' : '*ELR: Efficacy of Luminous Radiation (W/lm)*'}

def set_up_df_legend(keys):
    cpt = st.beta_expander('Table legend')
    for key in keys:
        cpt.markdown(legend_dict[key])  
        
def setup_tm30_report_info():
    expdr_info = st.sidebar.beta_expander('Set additional info for report')
    info = {'manufacturer' : expdr_info.text_input('Manufacturer',''),
            'date' :  expdr_info.text_input('Date',''),
            'model' : expdr_info.text_input('Model',''),
            'notes' : expdr_info.text_input('Notes','')}
    return info

def setup_colorimetric_info():
    st.sidebar.markdown("### Colorimetric options:")
    info = {'cieobs' : st.sidebar.selectbox('CIE observer',[x for x in lx._CMF['types'] if (x!='cie_std_dev_obs_f1')]),
            'relative_xyz' : st.sidebar.checkbox("Relative XYZ [Ymax=100]", True, key = 'relative_xyz')
            }
    return info


class Run:
    def __init__(self, option):
        
        # load option based settings:
        self.option = option
        self.opt = run_options[self.option][0] # get short name for option
        self.fcn = run_options[self.option][1] # select correct function to run based on option
        self.input_data_type = run_options[self.option][2]
        self.has_legend = run_options[self.option][3]
        self.title = run_options[self.option][4]
        self.data = None
        self.df_result = None # for storing results dataframe 
        self.info = {'info':None}
        
    def load_data(self):
        """ Load data"""
        # load and process spectra data:
        if self.input_data_type == 'spd':
            self.spectra_df, self.file_details = load_spectral_data()
            display_spectral_input_data(self.spectra_df, self.file_details)
            self.names = list(self.spectra_df.columns[1:]) if not self.has_legend else (['all'] + list(self.spectra_df.columns[1:])) # use has_legend to determine if requested calculation can handle more than 1 spd
            self.name = st.sidebar.selectbox('Select spectrum',self.names) # create spd selector, i.e. data indentifier ('all' or column name in dataframe)
            if self.name != 'all':
                spd_opts = ['all'] + list(self.spectra_df.columns[1:])
                index = spd_opts.index(self.name) - 1
                self.data = self.spectra_df.values.T[[0,index+1],:]
                self.names = [self.spectra_df.columns[1:][index]]
            else:
                self.data = self.spectra_df.values.T
                self.names = self.spectra_df.columns[1:]
        elif self.input_data_type == 'lid':
            self.data, self.file_details = load_LID_data()
            self.name = self.file_details['FileName']
            self.names = [self.name]
    
    def setup_info_section(self):
        if self.opt == 'tm30_report':
            self.info = setup_tm30_report_info() 
        else:
            if self.opt in ('colorimetric_quants', 'cies2036_quants'):
                self.info = setup_colorimetric_info()
  
    def run(self):
        """ Run requested operation """
        # calculate and display results:
        st.markdown(self.title)
        
        if self.opt == 'tm30_report':
            self.df_result, self.legend, tmp = self.fcn(self.data, self.names, **self.info)
            st.pyplot(tmp['fig'])
            st.text("To download image, right-click and select 'Save Image As ...'")
            
            st.markdown("""---""")
            st.dataframe(self.df_result)
            if self.has_legend: set_up_df_legend(self.legend)
        
        elif self.opt == 'lid_plots':
            self.df_result, self.legend, tmp = self.fcn(self.data, self.names)
            st.pyplot(tmp)
            if tmp is not None: st.text("To download image, right-click and select 'Save Image As ...'")
        
        else:
            self.df_result, self.legend, tmp = self.fcn(self.data, self.names, **self.info)
            st.dataframe(self.df_result)
            if self.has_legend: set_up_df_legend(self.legend)
            
        if self.df_result is not None:
            st.markdown("""---""")
            st.markdown(get_table_download_link_csv(self.df_result), unsafe_allow_html=True)   

            
 
def setup_luxpy_info():
    st.sidebar.image(logo, width=300)
    st.sidebar.markdown('## **Online calculator for lighting and color science**')
    st.sidebar.markdown('Luxpy {:s}, App {:s}'.format(lx.__version__, __version__))
    link = 'Code: [github.com/ksmet1977/luxpy](http://github.com/ksmet1977/luxpy)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    st.sidebar.markdown('Code author: Prof. dr. K.A.G. Smet')
    
def setup_control_panel_main():
    st.sidebar.markdown("""---""")
    st.sidebar.title('Control panel')
    option = st.sidebar.selectbox("Run options", list(run_options.keys()))
    return option
  
def cite():
    st.markdown("""---""")
    st.markdown("If you use **LUXPY**, please cite the following tutorial paper published in LEUKOS:")
    st.markdown("**Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23.** DOI: [10.1080/15502724.2018.1518717](https://doi.org/10.1080/15502724.2018.1518717)")
    st.markdown("""---""")
    
def explain_usage():
    st.markdown('### Usage:')
    st.markdown(' 1. Select calculation option.')
    st.markdown(' 2. Load data (set) (and set data format options; default = csv in column format).')
    st.markdown(' 3. Press RUN.')
    

def main():
    start = True
    setup_luxpy_info()
    option = setup_control_panel_main()
    
    if option != '': 
        engine = Run(option)
        engine.load_data()
        engine.setup_info_section()
    
        st.sidebar.markdown("""---""")
        if engine.data is not None:
            if st.sidebar.button('RUN'):
                start = False
                engine.run()
        
        if start: st.text('Scroll down control panel...') 
    
    
    if start: explain_usage()
        
    cite()
    
    

def calculate(option, df, **kwargs):
    global start
    df_res = None
    if option == 'ANSI/IESTM30 graphic report':
        name = st.sidebar.selectbox('Select spectrum',df.columns[1:])
        index = list(df.columns[1:]).index(name)
        linfo = st.sidebar.beta_expander('Set additional info for report')
        manufacturer = linfo.text_input('Manufacturer','')
        date = linfo.text_input('Date','')
        model = linfo.text_input('Model','')
        notes = linfo.text_input('Notes','')
        data = df.values.T[[0,index+1],:]
    elif ((option == 'Alpha-opic quantities (CIE S026)') | 
         (option == 'ANSI/IESTM30 quantities') |
         (option == 'CIE 13.3-1995 Ra, Ri quantities') |
         (option == 'CIE 224:2017 Rf, Rfi quantities'),
         (option == "SPD->(X,Y,Z), (x,y), (u',v'), (CCT,Duv)")):
        spd_opts = ['all'] + list(df.columns[1:])
        name = placeholder_spdname.selectbox('Select spectrum',spd_opts)
        if name != 'all':
            index = spd_opts.index(name) - 1
            data = df.values.T[[0,index+1],:]
            names = [df.columns[1:][index]]
        else:
            data = df.values.T
            names = df.columns[1:]
        rfl = kwargs.get('rfl',None)
    else:
        data = df.values.T
    
    st.sidebar.markdown("""---""")
    if st.sidebar.button('RUN'):
        start = False
        if option == 'ANSI/IESTM30 graphic report':
            axs, results = lx.cri.plot_tm30_report(data, 
                                                source = name, 
                                                manufacturer = manufacturer,
                                                date = date,
                                                model = model,
                                                notes = notes,
                                                save_fig_name = None)
            # plt.sca(axs['fig'])
            # canvas = plt.gca().figure.canvas
            # canvas.draw()
            # data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            # buf = data.reshape(canvas.get_width_height()[::-1] + (3,))
            # w, h, d = buf.shape
            # image = Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )
            st.pyplot(axs['fig'])
            # st.markdown(get_image_download_link(image), unsafe_allow_html=True)
            st.text("To download image, right-click and select 'Save Image As ...'")
            
        elif option == 'ANSI/IESTM30 quantities':
            d = spd_to_tm30(data)
            LER = lx.spd_to_ler(d['St'])
            xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
            uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
            quants = ['CCT','Duv'] + ['x','y',"u'","v'",'LER'] + ['Rf', 'Rg']
            quants += ['Rcsh{:1.0f}'.format(i+1) for i in range(d['Rcshj'].shape[0])] 
            quants += ['Rhsh{:1.0f}'.format(i+1) for i in range(d['Rhshj'].shape[0])] 
            quants += ['Rfh{:1.0f}'.format(i+1) for i in range(d['Rfhj'].shape[0])]
            quants += ['Rf{:1.0f}'.format(i+1) for i in range(d['Rfi'].shape[0])]
            
            df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                             xy.T,uv.T,LER.T,
                                             d['Rf'], d['Rg'],
                                             d['Rcshj'],d['Rhshj'],d['Rfhj'],d['Rfi'])).T,
                                   columns = quants,
                                   index = names)
            
            st.markdown('**ANSI/IES TM30 Quantities (CCT, Duv, Rf,Rg, ...)**')
            st.dataframe(df_res)
            cpt = st.beta_expander('Table legend')
            cpt.markdown('*CCT: Correlated Color Temperature (K)*')
            cpt.markdown('*Duv: distance from Planckian locus*')
            cpt.markdown('*xy: CIE 1931 2° xy chromaticity coordinates of illuminant white point*')
            cpt.markdown("*u'v': CIE 1976 2° u'v' chromaticity coordinates*")
            cpt.markdown('*LER: Luminous Efficacy of Radiation (lm/W)*')
            cpt.markdown('*Rf: general color fidelity index*')
            cpt.markdown('*Rg: gamut area index*')
            cpt.markdown('*Rcshj: local chroma shift for hue bin j*')
            cpt.markdown('*Rhshj: local hue shift for hue bin j*')
            cpt.markdown('*Rfhj: local color fidelity index for hue bin j*')
            cpt.markdown('*Rfi: specific color fidelity index for sample i*')
        
        elif option == 'CIE 13.3-1995 Ra, Ri quantities':
            d = spd_to_tm30(data)
            LER = lx.spd_to_ler(d['St'])
            Ra, _ = lx.cri.spd_to_cri(d['St'], cri_type = 'ciera', out = 'Rf,Rfi')
            _, Ri = lx.cri.spd_to_cri(d['St'], cri_type = 'ciera-14', out = 'Rf,Rfi')

            xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
            uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
            quants = ['CCT','Duv'] + ['x','y',"u'","v'",'LER'] + ['Ra']
            quants += ['R{:1.0f}'.format(i+1) for i in range(Ri.shape[0])]
            
            df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                             xy.T,uv.T,LER.T,
                                             Ra, Ri)).T,
                                   columns = quants,
                                   index = names)
            
            st.markdown('**CIE 13.3-1995 Ra, Ri quantities**')
            st.dataframe(df_res)
            cpt = st.beta_expander('Table legend')
            cpt.markdown('*CCT: Correlated Color Temperature (K)*')
            cpt.markdown('*Duv: distance from Planckian locus*')
            cpt.markdown('*xy: CIE 1931 2° xy chromaticity coordinates of illuminant white point*')
            cpt.markdown("*u'v': CIE 1976 2° u'v' chromaticity coordinates*")
            cpt.markdown('*LER: Luminous Efficacy of Radiation (lm/W)*')
            cpt.markdown('*Ra: general color fidelity index*')
            cpt.markdown('*Ri: specific color fidelity index for sample i*')
  
        elif option == 'CIE 224:2017 Rf, Rfi quantities':
            d = spd_to_tm30(data)
            LER = lx.spd_to_ler(d['St'])
            Ra,Ri = lx.cri.spd_to_cierf(d['St'], out = 'Rf,Rfi')
            xy = lx.xyz_to_Yxy(d['xyztw_cct'])[...,1:]
            uv = lx.xyz_to_Yuv(d['xyztw_cct'])[...,1:]
            quants = ['CCT','Duv'] + ['x','y',"u'","v'",'LER'] + ['Rf']
            quants += ['Rf{:1.0f}'.format(i+1) for i in range(Ri.shape[0])]
            
            df_res = pd.DataFrame(np.vstack((d['cct'].T,d['duv'].T,
                                             xy.T,uv.T, LER.T,
                                             Ra, Ri)).T,
                                   columns = quants,
                                   index = names)
            
            st.markdown('**CIE 224:2017 Rf, Rfi quantities**')
            st.dataframe(df_res)
            cpt = st.beta_expander('Table legend')
            cpt.markdown('*CCT: Correlated Color Temperature (K)*')
            cpt.markdown('*Duv: distance from Planckian locus*')
            cpt.markdown('*xy: CIE 1931 2° xy chromaticity coordinates of illuminant white point*')
            cpt.markdown("*u'v': CIE 1976 2° u'v' chromaticity coordinates*")
            cpt.markdown('*LER: Luminous Efficacy of Radiation (lm/W)*')
            cpt.markdown('*Rf: general color fidelity index*')
            cpt.markdown('*Rfi: specific color fidelity index for sample i*')

    
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
                cpt = st.beta_expander('Table legend')
                cpt.markdown('*Ee: irradiance (W/m²)*')
                cpt.markdown('*EDI: Equivalent Daylight Illuminance (lux)*')
                cpt.markdown('*DER: Daylight Efficacy Ratio*')
                cpt.markdown('*ELR: Efficacy of Luminous Radiation (W/lm)*')
                
            except:
                st.markdown('Not implemented yet (03/05/2021)')
                
        elif option == "SPD->(X,Y,Z), (x,y), (u',v'), (CCT,Duv)":
            if rfl is not None:
                xyz, xyzw = lx.spd_to_xyz(data, cieobs = kwargs['cieobs'], relative = kwargs['relative_xyz'], rfl = kwargs['rfl'])
            else:
                xyz = lx.spd_to_xyz(data, cieobs = kwargs['cieobs'], relative = kwargs['relative_xyz'])
            cct, duv = lx.xyz_to_cct(xyz, out ='cct,duv')
            xy = lx.xyz_to_Yxy(xyz)[...,1:]
            uv = lx.xyz_to_Yuv(xyz)[...,1:]
            quants = ['X','Y','Z'] + ['x','y',"u'","v'"] + ['CCT','Duv']
            
            df_res = pd.DataFrame(np.vstack((xyz.T,
                                             xy.T,uv.T,
                                             cct.T, duv.T,
                                             )).T,
                                   columns = quants,
                                   index = names)
            
            st.markdown("**(X,Y,Z), (x,y), (u',v'), (CCT,Duv) for CIE observer {:s}**".format(kwargs['cieobs']))
            st.dataframe(df_res)
            cpt = st.beta_expander('Table legend')
            cpt.markdown('*XYZ: CIE X,Y,Z tristimulus values*')
            cpt.markdown('*xy: CIE xy chromaticity coordinates*')
            cpt.markdown("*u'v': CIE 1976 u'v' chromaticity coordinates*")
            cpt.markdown('*CCT: Correlated Color Temperature (K)*')
            cpt.markdown('*Duv: distance from Planckian locus*')

    else:
        start = False
        data = []
    return data, start, df_res
 
    


start = True           
            
def mainold():
    global start 
    df_download = None
    st.sidebar.image(logo, width=300)
    st.sidebar.markdown('## **Online calculator for lighting and color science**')
    st.sidebar.markdown('Luxpy {:s}, App {:s}'.format(lx.__version__, __version__))
    link = 'Code: [github.com/ksmet1977/luxpy](http://github.com/ksmet1977/luxpy)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    st.sidebar.markdown('Code author: Prof. dr. K.A.G. Smet')
    st.sidebar.markdown("""---""")
    st.sidebar.title('Control panel')
    option = st.sidebar.selectbox("Calculation options", ('',
                                                          'ANSI/IESTM30 quantities', 
                                                          'ANSI/IESTM30 graphic report',
                                                          'CIE 13.3-1995 Ra, Ri quantities',
                                                          'CIE 224:2017 Rf, Rfi quantities',
                                                          'Alpha-opic quantities (CIE S026)',
                                                          "SPD->(X,Y,Z), (x,y), (u',v'), (CCT,Duv)",
                                                          'Plot Luminous Intensity Distribution (IES/LDT files)'
                                                          ))
    st.sidebar.markdown("""---""")
    if option in ('ANSI/IESTM30 quantities',
                  'ANSI/IESTM30 graphic report',
                  'CIE 13.3-1995 Ra, Ri quantities',
                  'CIE 224:2017 Rf, Rfi quantities',
                  'Alpha-opic quantities (CIE S026)'):
        df, file_details, display = load_spectral_data()
        display_spectral_input_data(df, file_details, display)
        data, start, df_download = calculate(option, df)
    elif (option == "SPD->(X,Y,Z), (x,y), (u',v'), (CCT,Duv)"):
        df, file_details, display = load_spectral_data()
        display_spectral_input_data(df, file_details, display)
        # st.sidebar.markdown("""---""")
        st.sidebar.markdown("### XYZ options:")
        cieobs = st.sidebar.selectbox('CIE observer',[x for x in lx._CMF['types'] if (x!='cie_std_dev_obs_f1')])
        relative_xyz = st.sidebar.checkbox("Relative XYZ [Ymax=100]", True, key = 'relative_xyz')
        data, start, df_download = calculate(option, df, cieobs = cieobs, relative_xyz = relative_xyz)
    elif option in ('Plot Luminous Intensity Distribution (IES/LDT files)',):
        lid_dict = load_LID_file()
        st.sidebar.markdown("""---""")
        if st.sidebar.button('RUN'):
            start = display_LID_file(lid_dict)
    else:
        df, file_details, display = None, {}, 'No'
     
    if (option != '') &  (start):
       st.text('Scroll down control panel...')

        
    if start:
        st.markdown('### Usage:')
        st.markdown(' 1. Select calculation option.')
        st.markdown(' 2. Load data (set) (and set data format options; default = csv in column format).')
        st.markdown(' 3. Press RUN.')
    
    if df_download is not None:
        st.markdown("""---""")
        st.markdown(get_table_download_link_csv(df_download), unsafe_allow_html=True)    
    st.markdown("""---""")
    st.markdown("If you use **LUXPY**, please cite the following tutorial paper published in LEUKOS:")
    st.markdown("**Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23.** DOI: [10.1080/15502724.2018.1518717](https://doi.org/10.1080/15502724.2018.1518717)")
    st.markdown("""---""")
    #start = False
    
if __name__ == '__main__':
    main()
    
    