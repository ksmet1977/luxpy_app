# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import base64
from io import BytesIO, StringIO
from PIL import Image
import copy


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
    file_details = {"FileName":'',"FileType":'',"FileSize":''}
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

def load_dataframe():
    # Set title for this sidebar section:
    st.sidebar.markdown("""---""")
    st.sidebar.markdown("### Load dataframe:")
    
    # expander with data format options:
    expdr_dopts = st.sidebar.beta_expander("Data-format options")
    expdr_dopts.checkbox("Column format", True, key = 'options')
    header = 'infer' if expdr_dopts.checkbox("Data file has header", False, key = 'header') else None
    index_col = 0 if expdr_dopts.checkbox("First Column is Index", False, key = 'col_index') else None
   
    sep = expdr_dopts.selectbox('Separator',[',','\t',';'])
    
    # expander with data loading:
    expdr_dload = st.sidebar.beta_expander("Upload DataFrame csv file",True)
    uploaded_file = expdr_dload.file_uploader("",accept_multiple_files=False,type=['csv','dat','txt'])
    file_details = {"FileName":'',"FileType":'',"FileSize":''}
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        df = pd.read_csv(uploaded_file, header =  header, sep = sep, index_col = index_col) # read in data
        names = df.columns if (header == 'infer') else ['C{:1.0f}'.format(i+1) for i in range(len(df.columns))]
        df.columns = names
        print('df',df)
    else:
        df = pd.DataFrame(np.array([[100.0,100.0,100.0]]), index = ['EEW']) # D65 default
        names = ['X','Y','Z']
        df.columns = ['X','Y','Z'] 
        file_details['FileName'] = 'EEW (hard-coded)'
    return df, file_details

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
        
placeholder_indexselector = None
def display_dataframe(df, file_details, sidebar = True):
    st.sidebar.markdown('### Input dataframe:')
    expdr_dshow = st.sidebar.beta_expander('Show input dataframe') if sidebar  else st.beta_expander('Show input dataframe')      
    expdr_dshow.write(file_details)
    expdr_dshow.dataframe(df)

   

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
# Run classes, functions and variables
#------------------------------------------------------------------------------
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
    code = """
    import luxpy as lx                               # imports the luxpy package 
    spd = lx.spd('spd_data_file.csv')                # returns a numpy array with spectral data in csv-file with filename 'spd_data_file.csv'
    Rf =  lx.cri.spd_to_iesrf(spd)                   # returns the general color fidelity index as defined in ANSI/IES-TM30
    Rg =  lx.cri.spd_to_iesrg(spd)                   # returns the color gamut index as defined in ANSI/IES-TM30
    tm30_dict = lx.cri._tm30_process_spd(spd)        # returns a dictionary with tm30 quantities:  'Rf', 'Rg', 'Rfi', 'Rcshj', 'Rhshj', 'Rfhj', CCT, Duv, ...  
    CCT, Duv = tm30_dict['cct'],tm30_dict['duv']     # get CCT and Duv for 1931 2° XYZ input (i.e. the one used to determine the reference illuminant) from dictionary

    ler = lx.spd_to_ler(spd, cieobs = '1931_2')                    # Luminous Efficacy of Radiation calculated with CIE 1931 2° Ybar (= CIE 1924 Vlambda) 
    XYZ = lx.spd_to_xyz(spd, cieobs = '1931_2', relative = True)   # relative XYZ tristimulus values calculated with CIE 1931 2° observer
    Yxy = lx.xyz_to_Yxy(XYZ)                                       #  CIE x, y chromaticity corrdinates
    Yuv = lx.xyz_to_Yuv(XYZ)                                       # CIE 1976 u', v' chromaticity coordidinates
    """
    return df_res, legend, code, d

def plot_tm30_report(data, names, **kwargs):
    source = kwargs.get('source','')
    manufacturer = kwargs.get('manufacturer','')
    date = kwargs.get('date','')
    model = kwargs.get('model','')
    notes = kwargs.get('notes','')
    df_res, legend, code, d = calc_tm30_quants(data, names)

    axs, results = lx.cri.plot_tm30_report(d, 
                                            source = source, 
                                            manufacturer = manufacturer,
                                            date = date,
                                            model = model,
                                            notes = notes,
                                            save_fig_name = None)
    code = """
    import luxpy as lx                            # imports the luxpy package 
    spds = lx.spd('spd_data_file.csv')            # returns a numpy array with spectral data in csv-file with filename 'spd_data_file.csv'
    spd = spds[[0, 1]]                            # select a single specific spd (e.g. 1st; index = row number + 1) from the array
    _, tm30_dict = lx.cri.plot_tm30_report(spd)   # generates report and returns a dictionary with tm30 quantities:  'Rf', 'Rg', 'Rfi', 'Rcshj', 'Rhshj', 'Rfhj', CCT, Duv, ...  
    Rf, Rg = tm30_dict['Rf'], tm30_dict['Rg']     # e.g. get color fidelity index Rf and color gamut area index Rg from tm30_dict
    """
    return df_res, legend, code, axs
    
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
    code = """
    import luxpy as lx                                                # imports the luxpy package 
    spd = lx.spd('spd_data_file.csv')                                 # returns a numpy array with spectral data in csv-file with filename 'spd_data_file.csv'
    Ra, Ri =  lx.cri.spd_to_ciera(spd, out = 'Rf,Rfi')                # returns the general (Ra) and specific (Ri) color rendering fidelity indices as defined in CIE13.3:1995
    ler = lx.spd_to_ler(spd, cieobs = '1931_2')                       # Luminous Efficacy of Radiation calculated with CIE 1931 2° Ybar (= CIE 1924 Vlambda) 
    XYZ = lx.spd_to_xyz(spd, cieobs = '1931_2', relative = True)      # relative XYZ tristimulus values calculated with CIE 1931 2° observer
    CCT, Duv = lx.xyz_to_cct(XYZ, cieobs = '1931_2', out ='cct,duv')  # CCT and Duv for 1931 2° XYZ input
    Yxy = lx.xyz_to_Yxy(XYZ)                                          # CIE x, y chromaticity corrdinates
    Yuv = lx.xyz_to_Yuv(XYZ)                                          # CIE 1976 u', v' chromaticity coordidinates
    """
    return df_res, legend, code, None

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
    code = """
    import luxpy as lx                                                # imports the luxpy package 
    spd = lx.spd('spd_data_file.csv')                                 # returns a numpy array with spectral data in csv-file with filename 'spd_data_file.csv'
    Rf, Rfi =  lx.cri.spd_to_cierf(spd, out = 'Rf,Rfi')               # returns the general (Rf) and specific (Rfi) color rendering fidelity indices as defined in CIE224:2017
    ler = lx.spd_to_ler(spd, cieobs = '1931_2')                       # Luminous Efficacy of Radiation calculated with CIE 1931 2° Ybar (= CIE 1924 Vlambda) 
    XYZ = lx.spd_to_xyz(spd, cieobs = '1931_2', relative = True)      # relative XYZ tristimulus values calculated with CIE 1931 2° observer
    CCT, Duv = lx.xyz_to_cct(XYZ, cieobs = '1931_2', out ='cct,duv')  # CCT and Duv for 1931 2° XYZ input
    Yxy = lx.xyz_to_Yxy(XYZ)                                          # CIE x, y chromaticity corrdinates
    Yuv = lx.xyz_to_Yuv(XYZ)                                          # CIE 1976 u', v' chromaticity coordidinates
    """
    return df_res, legend, code, None
    
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
    code = """
    import luxpy as lx                  # imports the luxpy package 
    spd = lx.spd('spd_data_file.csv')   # returns a numpy array with spectral data in csv-file with filename 'spd_data_file.csv'
    aEe = ph.spd_to_aopicE(spd)         # returns a numpy array with the alpha-opic irradiance 
    aedi = ph.spd_to_aopicEDI(spd,cieobs = '1931_2') # alpha-opic Equivalent Daylight Illuminance with illuminance Ev calculated using the Ybar function in the CIE 1931 2° observer
    ader = ph.spd_to_aopicDER(spd,cieobs = '1931_2') # alpha-opic Daylight Efficacy Ratio with illuminance Ev calculated using the Ybar function in the CIE 1931 2° observer
    aelr = ph.spd_to_aopicELR(spd,cieobs = '1931_2') # alpha-opic Efficacy of Luminous Radiation with illuminance Ev calculated using the Ybar function in the CIE 1931 2° observer
    """
    return df_res, legend, code, None

def calc_colorimetric_quants(data, names, **kwargs):
    rfl = kwargs.get('rfl',None)
    xyz, xyzw = lx.spd_to_xyz(data, cieobs = kwargs['cieobs'], relative = kwargs['relative'], rfl = kwargs.get('rfl',None), out = 2)
    if rfl is not None: xyz = xyz[:,0,:] # get rid of light source dimension
    cct, duv = lx.xyz_to_cct(xyz, out ='cct,duv', cieobs = kwargs['cieobs'])
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
    code = """
    import luxpy as lx                                                # imports the luxpy package 
    spd = lx.spd('spd_data_file.csv')                                 # returns a numpy array with spectral data in csv-file with filename 'spd_data_file.csv'
    XYZ = lx.spd_to_xyz(spd, cieobs = '1931_2', relative = True)      #  relative XYZ tristimulus values calculated with CIE 1931 2° observer
    CCT, Duv = lx.xyz_to_cct(XYZ, cieobs = '1931_2', out ='cct,duv')  # CCT and Duv for 1931 2° XYZ input
    Yxy = lx.xyz_to_Yxy(XYZ)                                          # CIE x, y chromaticity corrdinates
    Yuv = lx.xyz_to_Yuv(XYZ)                                          # CIE 1976 u', v' chromaticity coordidinates
    """
    return df_res, legend, code, xyzw

def plot_ies_ldt_lid(LID, names, **kwargs):
    fig = generate_LID_plots(LID)
    
    code = """ 
    import luxpy.toolboxes.iolidfiles as iolid          # imports the iolidfiles toolbox from the luxpy package
    LID = iolid.read_lamp_data('lid_data_file.ies')     # returns a dictionary with the info stored in the IES (or LDT) file named 'lid_data_file.ies'
    iolid.draw_lid(LID)                                 # generate a figure with a polar of the C0-C180 and C90-C270 planes
    iolid.render_lid(LID)                               # generate a figure with a 1-bounce physical based render of the LID in a simple scene composed of a Lambertian wall and a floor
    """
    
    return (None, None, code, fig)

def custom_code(data, names, code, **kwargs):
    __tmp__ = {}
    __legend__ = None # for legend
    __results__ = None
    __code__ = copy.copy(code) # make a copy so this variable is not accidently overwritten
    
    # prepare code for execution:
    indent = '    '
    
    if '__results__' not in code: code = code + "\n__results__ = '!!! No __results__ variable with output was defined in user code !!!'\n"
    return_string = "return __results__, __legend__\n"  if '__legend__' in code else "return __results__\n" 
    
    code = ("def __user_code__(data, names, **kwargs):\n" + \
           '\n'.join([indent + line for line in code.split('\n')]) + \
           return_string)
        
    # execute user defined code: 
    exec(code, __tmp__)
    if ('__results__' in code) & ('__legend__' in code): 
       __results__, __legend__ = __tmp__['__user_code__'](data,names,**kwargs)
    elif ('__results__' in code):
       __results__ = __tmp__['__user_code__'](data,names,**kwargs)

         
    # update legend_dict 
    if __legend__ is not None: 
        global legend_dict 
        legend_dict.update(__legend__)
        __legend__ = list(__legend__.keys())
    return __results__, __legend__, __code__, None 

    

# run options --> {option : (short name, function, input datatype, has_legend, title)}
run_options = {'' : ('', None, None, False, ''),
                'ANSI/IESTM30 graphic report' : ('tm30_report', plot_tm30_report, 'spd', False, '**ANSI/IESTM30 graphic report**'),
               'ANSI/IESTM30 quantities' : ('tm30_quants' ,calc_tm30_quants, 'spd', True, '**ANSI/IES TM30 Quantities (CCT, Duv, Rf,Rg, ...)**'),
               'CIE 13.3-1995 Ra, Ri quantities' : ('ciera', calc_ciera_quants, 'spd',True,'**CIE 13.3-1995 Ra, Ri quantities**'),
               'CIE 224:2017 Rf, Rfi quantities': ('cierf', calc_cierf_quants, 'spd', True,'**CIE 224:2017 Rf, Rfi quantities**'),
               "SPD->(X,Y,Z), (x,y), (u',v'), (CCT,Duv)" :('colorimetric_quants',calc_colorimetric_quants, 'spd', True,"**Colorimetric quantities: (X,Y,Z), (x,y), (u',v'), (CCT,Duv)**"),
               'Alpha-opic quantities (CIE S026)' : ('cies2036_quants', calc_cies026_quants, 'spd', True, 'Alpha-opic quantities (CIE S026)'),
               'Plot Luminous Intensity Distribution (IES/LDT files)' : ('lid_plots', plot_ies_ldt_lid, 'lid', False, '**Luminous Intensity Distiribution (polar plot and render)**'),
               'Write your own code' : ('custom_code', custom_code, ('spd','lid','general'), True, '**Output of user generated code**')
               }

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
            'relative' : st.sidebar.checkbox("Relative XYZ [Ymax=100]", True, key = 'relative')
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
        self.code_example = None
        self.user_code = None
        if self.opt == 'custom_code':
            st.markdown("**Write and run your own Luxpy code**")
            st.markdown("*Don't know how? Have a look at the FREE (Open Access) [tutorial paper in LEUKOS](https://doi.org/10.1080/15502724.2018.1518717)*")
            ccode_expdr = st.beta_expander('!!! READ ME !!!')
            ccode_expdr.text("Write your own code in the field below.")
            ccode_expdr.text("""
                             - Uploaded data is available in the variable `data`.
                             - Identifying 'names' for the data are available in the variable `names`.
                             - Additional user defined arguments (when available),
                               such as e.g. `cieobs` and `relative`, are stored as keys
                               in a dictionary `kwargs`.
                               (e.g. to access cieobs use `cieobs = kwargs['cieobs'].)
                             - Final output must be stored in a tuple called `__results__`.
                             - Final output can be a pandas dataframe or a figure handle to a plot
                             - Optionally, a 'legend' to the column names or index names used in
                               the dataframe can be defined in a dictionary `__legend__`of the form:
                                   {'legend key' : 'explanation string'}
                               (e.g. __legend__ = {'XYZ' : 'XYZ tristimulus values',
                                                   'CCT' : 'Correlated Color Temperature (K)',
                                                   }
                            """)
            ccode_expdr.text(" For example, see default code in text field")

            
            
            
        
    def load_data(self):
        """ Load data"""
        # load and process spectra data:
        if isinstance(self.input_data_type,tuple):
            self.input_data_type = st.sidebar.selectbox("Input data type",self.input_data_type)
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
        elif self.input_data_type == 'general':
            self.df, self.file_details = load_dataframe()
            display_dataframe(self.df, self.file_details)
            self.name = self.file_details['FileName']
            self.names = list(self.df.index)
            self.columnnames = self.df.columns
            self.data = self.df.values
    
    def setup_info_section(self):
        if self.opt == 'tm30_report':
            self.info = setup_tm30_report_info() 
       
        elif self.opt == 'custom_code':
            self.info = setup_colorimetric_info()
            
            self.code_example_spd = \
"""import luxpy, pandas, numpy, matplotlib.pyplot

XYZ = luxpy.spd_to_xyz(data, cieobs = kwargs['cieobs'], relative = kwargs['relative'])
CCT, Duv = luxpy.xyz_to_cct(XYZ, cieobs = kwargs['cieobs'], out = 'cct,duv')
Yxy = luxpy.xyz_to_Yxy(XYZ)

fig, ax = matplotlib.pyplot.subplots(1,2, figsize=(8,4))
for i in range(data.shape[0]-1):
    ax[0].plot(data[0],data[i+1],label = names[i])
    ax[0].legend()
    ax[0].set_xlabel('Wavelength (nm)')
    ax[0].set_ylabel('Intentensity (W/m².sr.nm)')
    luxpy.plotSL(cspace = 'Yxy', cieobs = kwargs['cieobs'], axh = ax[1], diagram_colors = True)
    ax[1].plot(Yxy[i,1],Yxy[i,2],'o', markeredgecolor = 'k', label = names[i])
    ax[1].legend()

df = pandas.DataFrame(numpy.vstack((XYZ.T, CCT.T, Duv.T, Yxy[...,1:].T)).T, 
                      columns = ['X', 'Y', 'Z', 'CCT', 'Duv', 'x', 'y'], 
                      index = names)

__results__ = (fig, df)
__legend__ = {'XYZ' : '*XYZ: XYZ tristimulus values*',
          'CCT' : '*CCT: Correlated Color Temperature (K)*',
          'Duv' : '*Duv: Distance to blackbody locus in CIE 1960 uv diagram*',
          'xy'  : '*xy: CIE x,y chromaticity coordinates*'
          }
"""

            self.code_example_basic_spd = \
"""import luxpy, pandas, numpy
XYZ = luxpy.spd_to_xyz(data, cieobs = kwargs['cieobs'], relative = kwargs['relative'])
CCT, Duv = luxpy.xyz_to_cct(XYZ, cieobs = kwargs['cieobs'], out = 'cct,duv')
Yxy = luxpy.xyz_to_Yxy(XYZ)
__results__ = pandas.DataFrame(numpy.vstack((XYZ.T, CCT.T, Duv.T, Yxy[...,1:].T)).T, 
                               columns = ['X', 'Y', 'Z', 'CCT', 'Duv', 'x', 'y'], 
                               index = names) 
"""

            self.code_example_lid=\
"""import luxpy, pandas, numpy, matplotlib.pyplot
from luxpy.toolboxes import iolidfiles as iolid
fig = matplotlib.pyplot.figure(figsize=[10,3])
axs = [fig.add_subplot(131, projection = 'polar'), 
       fig.add_subplot(132, projection = '3d'),
       fig.add_subplot(133)]
iolid.draw_lid(data, ax = axs[0], polar_plot_Cx_planes = [0,45,90])
iolid.draw_lid(data, ax = axs[1], projection = '3d')
iolid.render_lid(data, ax2D = axs[2], ax3D = False, sensor_resolution = 50)
matplotlib.pyplot.tight_layout()
__results__ = fig
"""

            self.code_example_basic_lid=\
"""import luxpy, pandas, numpy, matplotlib.pyplot
from luxpy.toolboxes import iolidfiles as iolid
fig = matplotlib.pyplot.figure(figsize=[10,4])
axs = [fig.add_subplot(121, projection = 'polar'), 
       fig.add_subplot(122, projection = '3d')]
iolid.draw_lid(data, ax = axs[0])
iolid.draw_lid(data, ax = axs[1], projection = '3d')
matplotlib.pyplot.tight_layout()
__results__ = fig
"""

            self.code_example_general = \
"""import luxpy, pandas, numpy,matplotlib.pyplot
XYZ = data
CCT, Duv = luxpy.xyz_to_cct(XYZ, cieobs = kwargs['cieobs'], out = 'cct,duv')
Yxy = luxpy.xyz_to_Yxy(XYZ)
Yuv = luxpy.xyz_to_Yuv(XYZ)
fig, ax = matplotlib.pyplot.subplots(1,2,figsize=(8,4))
for i in range(Yxy.shape[0]):
    luxpy.plotSL(cspace = 'Yxy', cieobs = kwargs['cieobs'], axh = ax[0], diagram_colors = True)
    ax[0].plot(Yxy[i,1],Yxy[i,2],'o', markeredgecolor = 'k',label = names[i])
    ax[0].legend()
    luxpy.plotSL(cspace = 'Yuv', cieobs = kwargs['cieobs'], axh = ax[1], diagram_colors = True)
    ax[1].plot(Yuv[i,1],Yuv[i,2],'o', markeredgecolor = 'k',label = names[i])
    ax[1].legend()
df = pandas.DataFrame(numpy.vstack((XYZ.T, CCT.T, Duv.T, Yxy[...,1:].T, Yuv[...,1:].T)).T, 
                               columns = ['X', 'Y', 'Z', 'CCT', 'Duv', 'x', 'y',"u'", "v'"], 
                               index = names) 
__results__ = (fig, df)
__legend__ = {'XYZ' : '*XYZ: XYZ tristimulus values*',
          'CCT' : '*CCT: Correlated Color Temperature (K)*',
          'Duv' : '*Duv: Distance to blackbody locus in CIE 1960 uv diagram*',
          'xy'  : '*xy: CIE x,y chromaticity coordinates*',
          "u'v"  : "*u'v': CIE 1976 u',v' chromaticity coordinates*"
          }
"""

            self.code_example_basic_general = \
"""import luxpy, pandas, numpy
XYZ = data
CCT, Duv = luxpy.xyz_to_cct(XYZ, cieobs = kwargs['cieobs'], out = 'cct,duv')
Yxy = luxpy.xyz_to_Yxy(XYZ)
Yuv = luxpy.xyz_to_Yxy(XYZ)
__results__ = pandas.DataFrame(numpy.vstack((XYZ.T, CCT.T, Duv.T, Yxy[...,1:].T, Yuv[...,1:].T)).T, 
                               columns = ['X', 'Y', 'Z', 'CCT', 'Duv', 'x', 'y',"u'", "v'"], 
                               index = names) 
"""

            if self.input_data_type == "spd":
                self.code_example = self.code_example_spd
                self.code_example_basic = self.code_example_basic_spd
            elif self.input_data_type == "lid":
                self.code_example = self.code_example_lid
                self.code_example_basic = self.code_example_basic_lid
            elif self.input_data_type == "general":
                self.code_example = self.code_example_general
                self.code_example_basic = self.code_example_basic_general
                
            ccode_expr_text_area = st.beta_expander("Enter user code here",True)
            self.code_example_is_basic = ccode_expr_text_area.checkbox('Show basic (no plots, no legend) code example',True)
            code_example = self.code_example_basic if self.code_example_is_basic else self.code_example
            text_area_height = 200 if self.code_example_is_basic else 300
            self.custom_code = ccode_expr_text_area.text_area("Press ctrl-enter to load code!!!", 
                                                         value = code_example, 
                                                         height = text_area_height)
            ccode_expr_code = st.beta_expander("Check user code with Python language highlighting",False)
            ccode_expr_code.code(self.custom_code)
        
        else:
            if self.opt in ('colorimetric_quants', 'cies2036_quants'):
                self.info = setup_colorimetric_info()
                

        
  
    def run(self):
        """ Run requested operation """
        # calculate and display results:
        st.markdown(self.title)
        
        if self.opt == 'tm30_report':
            self.df_result, self.legend, self.code_example, tmp = self.fcn(self.data, self.names, **self.info)
            st.pyplot(tmp['fig'])
            st.text("To download image, right-click and select 'Save Image As ...'")
            
            st.markdown("""---""")
            st.dataframe(self.df_result)
            if self.has_legend: set_up_df_legend(self.legend)
        
        elif self.opt == 'lid_plots':
            self.df_result, self.legend, self.code_example, tmp = self.fcn(self.data, self.names)
            st.pyplot(tmp)
            if tmp is not None: st.text("To download image, right-click and select 'Save Image As ...'")
        
        elif self.opt == 'custom_code':
            self.df_result, self.legend, self.code_example, tmp = self.fcn(self.data, self.names, self.custom_code, **self.info)
            
            # try and process results to screen output:
            if not isinstance(self.df_result,tuple): self.df_result = (self.df_result,)
            for result in self.df_result: 
                # st.markdown("""---""")
                if isinstance(result, plt.Figure): 
                    st.pyplot(result)
                    st.text("To download image, right-click and select 'Save Image As ...'")
                elif isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                    if self.legend is not None: set_up_df_legend(self.legend)
                    st.markdown(get_table_download_link_csv(result), unsafe_allow_html=True)
                else:
                    if result is not None: st.write(result) # try write
            
            
        else:
            self.df_result, self.legend, self.code_example, tmp = self.fcn(self.data, self.names, **self.info)
            st.dataframe(self.df_result)
            if self.has_legend: set_up_df_legend(self.legend)
            
        if (self.df_result is not None) & (self.opt != 'custom_code'):
            st.markdown("""---""")
            st.markdown(get_table_download_link_csv(self.df_result), unsafe_allow_html=True)   

        if (self.code_example is not None) & (self.opt != 'custom_code'):
            expdr_code = st.beta_expander('Luxpy Coding Tutorial: show simple code example that generates output',False)
            expdr_code.code(self.code_example)
 
def setup_luxpy_info():
    st.sidebar.image(logo, width=200)
    st.sidebar.markdown('## **Online calculator for lighting and color science**')
    st.sidebar.markdown('Luxpy {:s}, App {:s}'.format(lx.__version__, __version__))
    link = 'Code: [github.com/ksmet1977/luxpy](http://github.com/ksmet1977/luxpy)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    st.sidebar.markdown('Code author: Prof. dr. K.A.G. Smet')
    #st.sidebar.markdown('doi:[10.1080/15502724.2018.1518717](https://doi.org/10.1080/15502724.2018.1518717)')
    
def setup_control_panel_main():
    st.sidebar.markdown("""---""")
    st.sidebar.title('Control panel')
    option = st.sidebar.selectbox("Run options", list(run_options.keys()))
    return option
  
def cite():
    st.markdown("""---""")
    st.markdown("""If you use **LUXPY**, please cite the following tutorial paper published in LEUKOS:""")
    st.markdown("""**Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23.** DOI: [10.1080/15502724.2018.1518717](https://doi.org/10.1080/15502724.2018.1518717)""")
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
        
        if engine.opt == 'custom_code': start = False
        if start: st.text('Scroll down control panel...') 
        
    if start: explain_usage()
    
        
    cite()
    
    
    
    
  
if __name__ == '__main__':
    main()
    
    