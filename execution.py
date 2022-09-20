"""
Arthur Rohaert
Tue Dec 21 10:50:40 2021
"""

# %%------------------------------------------------------- Import the packages
###############################################################################

import os
import datetime
import numpy as np
import requests
import pandas as pd
from scipy.optimize import minimize
from plots import plot_detector_diagram, plot_detector_series, plot_data

# %%--------------------------------------------------------------- Open thread

def try_execution(gui, input_information):
    try:
        execution(gui,input_information)
    except NameError as name_error:
        print(f"What went wrong is {name_error}")
        gui.progress = 0
        gui.percentage = 0
        gui.ids.probar.value = 0
        gui.executing = False
        gui.ids.execute_button.text = "Try again"
        gui.ids.status.markup = True
        gui.ids.status.text = "[color=FF4D76]Something went wrong :([/color]"
        gui.progress_message = "Please revise your choices."

# %%-------------------------------------------------------------- calculations
###############################################################################

def execution(gui, input_information):
    # ---------------------------------------------------- establish input data
    download_data = input_information["download_data"]
    if download_data:
        dates = input_information['dates']
        detectors = input_information['detectors']
        username = input_information['username']
        password = input_information['password']
    wd = input_information["directory"]

    # ----------------------------------------------------------------- sign in
    if download_data:
        gui.update(0, "signing in...")
        # start session and log in
        s = requests.session()
        r = s.post("https://pems.dot.ca.gov/",
                    {"username": username,
                      "password": password, "login": "Login"})

        # verify log in
        ERROR = 'Incorrect username or password.'
        if ERROR in str(r.content):
            raise Exception(ERROR)

        # ----------------------------------------------- download the raw data
        nodf = 0 # number of downloaded files
        tnof = sum([len(values) for values in dates.values()])*len(detectors)*2
        gui.update(0, f"downloading files... ({nodf}/{tnof})")
        if not os.path.isdir(os.path.join(wd,"raw_data")):
            os.mkdir(os.path.join(wd,"raw_data"))

        ref_date = datetime.datetime.strptime('01/01/1970 00:00', '%m/%d/%Y %H:%M')
        for date_label, date_list in dates.items():
            for counter, date in enumerate(date_list):
                # split the string into a start date and an end date
                start_date, end_date = date.split('till')
                # convert to datetime objects
                start_date = datetime.datetime.strptime(start_date, '%m/%d/%Y%H:%M')
                end_date = datetime.datetime.strptime(end_date, '%m/%d/%Y%H:%M')
                # convert to timestamps
                start_id = 86400 * (start_date - ref_date).days + \
                    (start_date - ref_date).seconds
                end_id = 86400 * (end_date - ref_date).days + \
                    (end_date - ref_date).seconds

                for detector, det_id in detectors.items():

                    # construct links
                    url_1 = f'https://pems.dot.ca.gov/?report_form=1&dnode=VDS&content=loops&tab=det_timeseries&export=text&station_id={det_id:.0f}&s_time_id={start_id:.0f}&e_time_id={end_id:.0f}&tod=all&tod_from=0&tod_to=0&q=flow&q2=occ&gn=5min&agg=on&lane1=on&lane2=on&lane3=on&lane4=on&lane5=on&lane6=on&lane7=on&lane8=on&lane9=on&lane10=on'
                    url_2 = f'https://pems.dot.ca.gov/?report_form=1&dnode=VDS&content=loops&tab=det_timeseries&export=text&station_id={det_id:.0f}&s_time_id={start_id:.0f}&e_time_id={end_id:.0f}&tod=all&tod_from=0&tod_to=0&q=speed&q2=truck_flow&gn=5min&agg=on&lane1=on&lane2=on&lane3=on&lane4=on&lane5=on&lane6=on&lane7=on&lane8=on&lane9=on&lane10=on'

                    # download files
                    filename = date_label + '_' + str(counter) + '_' + detector
                    file_1 = s.get(url_1) # getting flow and occupancy
                    output = open(os.path.join(wd,"raw_data",filename+'_1.txt'), 'wb')
                    output.write(file_1.content)
                    output.close()

                    file_2 = s.get(url_2) # getting speed and truck flow
                    output = open(os.path.join(wd,"raw_data",filename+'_2.txt'), 'wb')
                    output.write(file_2.content)
                    output.close()

                    nodf += 2 # two files at once
                    gui.update(270*2/tnof/2, f"downloading files... ({nodf}/{tnof})")

        # ------------------------------------------------ process the raw data
        nopf = 0 # number of processed files
        gui.update(0, f"processing files... ({nopf}/{tnof})")
        dataframes_list = []
        for date_label, date_list in dates.items():
            for count, date in enumerate(date_list):
                for detector in detectors.keys():
                    filename = date_label + '_' + str(count) + '_' + detector
                    df = extract_data(wd, filename)
                    df['Scenario'] = date_label
                    df['Detector'] = detector
                    dataframes_list.append(df)
                    nopf += 2 # number of processed files
                    gui.update(270*2/tnof/2, f"processing files... ({nopf}/{tnof})")

        gui.update(0, "merging data... ")
        data = pd.concat(dataframes_list)
        data = data.set_index([data.index, 'Detector', 'Scenario'])
        data = data.sort_index(axis=1, key=lambda x: x.str.lower())
        data.to_excel(os.path.join(wd,'data.xlsx'))

    else:
        gui.update(0, status_update="loading local dataset")
        xls_evac = pd.ExcelFile(input_information["file"])
        data = pd.read_excel(xls_evac,index_col=(0,1,2))
        gui.update(30)
        detectors, dates = get_indexes_and_dates(data)

# ----------------------------------------------- plot individual relationships
    if input_information["generate_plots"][0]:
        nopp = 0 # number of produced plots
        tnop = len(detectors.keys())*len(dates.keys())
        gui.update(0, f"producing scatterplots... (0/{tnop})")
        for detector in detectors.keys():
            for date_label in dates.keys():
                title = f'{date_label} - detector {detector} (VDS {detectors[detector]})'
                (data.loc[:,detector,date_label]).head()
                plot_detector_diagram(title, data.loc[:,detector,date_label], input_information)

                nopp += 1
                gui.update(200/tnop, f"producing scatterplots... ({nopp}/{tnop})")


# ------------------------------------------------------------ plot time series
    if input_information["generate_plots"][1]:
        nopp = 0 # number of produced plots
        tnop = len(detectors.keys())*sum([len(values) for values in dates.values()])
        gui.update(0, f"producing time series... (0/{tnop})")
        data = data.sort_values(by=['Detector','Time'],ascending=[True,True])
        for detector in detectors.keys():
            for date_label in dates.keys():
                for count, period in enumerate(dates[date_label]):
                    start_date, end_date = period.split('till')
                    start_date = datetime.datetime.strptime(start_date, '%m/%d/%Y%H:%M')
                    end_date = datetime.datetime.strptime(end_date, '%m/%d/%Y%H:%M')
                    df = data.loc[:,detector,date_label][start_date:end_date]
                    title = f'{date_label} ({count}) - detector {detector} (VDS {detectors[detector]})'
                    plot_detector_series(title, df, input_information)

                    nopp += 1
                    gui.update(200/tnop, f"producing time series... ({nopp}/{tnop})")

    # ----------------------------------- plot relationships for entire dataset
    if input_information["generate_plots"][2]:
        gui.update(0, "producing a scatter plot for the entire dataset")
        plot_data(input_information, data, title = 'Routine and evacuation diagram')
        gui.update(30)

    # --------------------------------------------------------------- plot fits
    if input_information["generate_plots"][3]:
        # establish models

        greenshield = {
            'name': 'Greenshield et al. (1935)',
            'parameters': ('vf', 'kj'),
            'function': lambda k,vf,kj:vf*(1-np.where(k<=0,0, np.where(k>kj,kj,k))/kj),
            'initial parameters': (110, 60),
            'keypoints' : [1]}

        underwood = {
            'name': 'Underwood (1961)',
            'parameters': ('vf', 'kc'),
            'function': lambda k, vf, kc : vf*np.exp(-np.where(k<=0,0,k)/kc),
            'initial parameters': (110, 20)}

        drake = {
            'name': 'Drake et al. (1967)',
            'parameters': ('vf', 'kc'),
            'function': lambda k, vf, kc : vf*np.exp(-0.5*(np.where(k<=0,0,k)/kc)**2),
            'initial parameters': (110, 20)}

        daganzo = {
            'name': 'Daganzo (1994)',
            'parameters': ('vf', 'kc', 'kj'),
            'function': lambda k,vf,kc,kj:np.where(k>=kj,0,np.where(k<=kc,vf,
                                                   vf*(kc/k)*(kj-k)/(kj-kc))),
            'initial parameters': (110, 20, 60),
            'keypoints' : [1,2]}

        def function_van_aerde(k, a, b, c, vf):
            """ This function calculates the vehicle speeds, based on the densities. It
            employs a linear iterpolation rather than solving the non linear problem.
            The distance between the velocity points is about 0.1 km/h. """
            v_list = np.linspace(vf-0.1, 0, round(10*vf))
            k_list = np.where(v_list>=vf,0,1/(a+b/(vf-v_list)+c*v_list))
            return np.interp(k, k_list, v_list)

        def parameters_van_aerde(vf, vc, kc, kj):
            """ This function calculates the parameters a,b,c,vf which are employed in
            the van aerde model, from the more phyiscal parameters vf, vc, kc, kj."""
            a = vf*(2*vc-vf)/(kj*vc**2)
            b = vf*(vf-vc)**2/(kj*vc**2)
            c = (1/(vc*kc))-vf/(kj*vc**2)
            return a,b,c,vf

        van_aerde = {
            'name': 'Van Aerde & Rakha (1995)',
            'parameters': ('a','b','c','vf'),
            'function': function_van_aerde,
            'initial parameters': parameters_van_aerde(110, 80, 20, 60)}

        del_castillo  = {
            'name': 'Del Castillo & Benítez (1995)',
            'parameters': ('vf', 'kj','c'),
            'function': lambda k, vf, kj, c: np.where(k>=kj,0,vf*(1-np.exp((c/vf)*(1-(kj/k))))),
            'initial parameters': (105, 65, 46)}

        models = [greenshield, underwood, drake,
                  daganzo, van_aerde, del_castillo]

        # ------------------------------------------------ fitting and plotting
        for index, model in enumerate(models):
            if not input_information["fit_models"][index]: continue
            gui.update(0, f"fitting and plotting {model['name']}")
            model = fit_data(data, model)
            plot_data(input_information, data, model, model['name'])
            gui.update(50)

        # ------------------------------ saving all parameters in an excel file
        gui.update(0, "saving optimal parameter values")
        p = {}
        scenarios = set(data.index.get_level_values(2))
        for scenario in scenarios:
            for index, model in enumerate(models):
                if not input_information["fit_models"][index]: continue
                p[model['name'] + ' - ' + scenario] = dict(zip(model['parameters'], model['optimal parameters ('+scenario.lower()+')']))
        p = pd.DataFrame.from_dict(p)
        p.to_excel(os.path.join(wd,'fitting parameters.xlsx'))

    # -------------------------------------------------------- Wrapping this up
    gui.update(1000, "done! :)")
    gui.progress_message = "Progress: 100%"
    gui.ids.execute_button.text = "Execute again"
    gui.executing = False

# %%------------------------------------------------------------ Help functions
###############################################################################

def extract_data(wd, filename):
    """
    Extract the data from the spreadsheets.
    Converts the data into usefull, metric units.
    Calculates the traffic density and the average vehichle length.
    Returns the data in a panda dataframe.

    Parameters
    ----------
    wd : the path of the directory (string)
    filename : the filename of the data (string)

    Returns
    -------
    df : the processed data (pandas dataframe)
    """

    # --------------------------------------------- reading the detectors data
    if not isinstance(filename, str):
        raise Exception('The filename needs to be a string')

    path_1 = os.path.join(wd,"raw_data",filename+'_1.txt')
    path_2 = os.path.join(wd,"raw_data",filename+'_2.txt')
    df_1 = pd.read_csv(path_1, sep='\t', index_col = 0)
    df_2 = pd.read_csv(path_2, sep='\t', index_col = 0)

    # ------------------------------------------------ formatting the dataframe
    # merge the two excel sheets
    df = pd.merge(df_1, df_2, left_index=True, right_index=True)

    # remove the double columns
    nol = df['# Lane Points_x'].values[0]
    df = df.drop(['# Lane Points_x','% Observed_x','# Lane Points_y'], axis=1)

    # rename columns
    lanes = [f'L{i+1}' for i in range(nol)]
    list.append(lanes, 'T')
    old_lanes = [f'Lane {i+1} ' for i in range(nol)]
    list.append(old_lanes, '')

    rename = {}
    for i in range(nol+1):
        rename[old_lanes[i] + 'Flow (Veh/5 Minutes)'] = \
            lanes[i] + ' Flow (veh/h/lane)'
        rename[old_lanes[i] + 'Occ (%)'] = lanes[i] + ' Occupancy (-)'
        rename[old_lanes[i] + 'Speed (mph)'] = lanes[i] + ' Speed (km/h)'
        rename[old_lanes[i] + 'Truck Flow (Veh/5 Minutes)'] = \
            lanes[i] + ' Truck Flow (trucks/h/lane)'
    rename['% Observed_y'] = 'Detector health (%)'
    rename['Occupancy (%)'] = lanes[i] + ' Occupancy (-)'
    df = df.rename(columns=rename)

    # reorder columns
    df = df.sort_index(axis=1, key=lambda x: x.str.lower())

    # ---------------------------------------------- Convert imperial to metric
    # speed: miles per hour -> kilometers per hour
    mi_to_km = lambda x : x*1.609344
    cols = [lane + ' Speed (km/h)' for lane in lanes]
    df[cols] = mi_to_km(df[cols]) #*mi_to_km

    # flow: per timestep -> per hour
    if df.index.name == '5 Minutes':
        samples_per_hour = 60/5 # samples per hour
        df.index.names = ['Time']
    else:
        raise Exception('No conversion defined for this granularity')
    cols = [lane + ' Flow (veh/h/lane)' for lane in lanes]
    cols.extend([lane + ' Truck Flow (trucks/h/lane)' for lane in lanes])
    df[cols] = df[cols]*samples_per_hour

    # total flow -> per lane
    df['T Flow (veh/h/lane)'] = df['T Flow (veh/h/lane)']/nol
    df['T Truck Flow (trucks/h/lane)'] = \
    df['T Truck Flow (trucks/h/lane)']/nol

    # occupancy: percentage -> decimal
    cols = [lane + ' Occupancy (-)' for lane in lanes]
    df[cols] = df[cols]*1E-2

    # ------------------ calculating traffic density and average vehicle length
    for lane in lanes:
        # traffic dansity
        df[lane +' Density (veh/km/lane)'] = \
        df[lane + ' Flow (veh/h/lane)'] / df[lane + ' Speed (km/h)']
        # average vehicle length
        df[lane +' Average vehicle length (m/veh)'] = 1000 * \
        df[lane + ' Speed (km/h)'] * df[lane + ' Occupancy (-)'] /\
        df[lane + ' Flow (veh/h/lane)']

    df.index = pd.to_datetime(df.index)
    return df

# %% ------------------------------------------------------ Auxiliary functions
###############################################################################

def get_indexes_and_dates(data):
    """ This function reads the indexes of the dataframe to extract all
    detectors and dates """
    detectors = list(set(data.index.get_level_values("Detector")))
    detectors.sort()

    scenarios = set(data.index.get_level_values("Scenario"))

    dates = {}
    for scenario in scenarios:
        timelist = list((data.loc[:,detectors[0],scenario].index.get_level_values("Time")))
        periods = []
        begin = timelist[0]
        for i in range(1,len(timelist)):
            if ((timelist[i] - timelist[i-1]).seconds > 300) or ((timelist[i] - timelist[i-1]).days > 0):
                end = timelist[i-1]
                periods.append(begin.strftime("%m/%d/%Y%H:%M")+"till"+end.strftime("%m/%d/%Y%H:%M"))
                begin = timelist[i]
        end = timelist[-1]
        periods.append(begin.strftime("%m/%d/%Y%H:%M")+"till"+end.strftime("%m/%d/%Y%H:%M"))
        dates[scenario] = periods
    detectors = {key:0 for key in detectors}
    return detectors, dates

def least_squares_error(parameters, df, model):
    """ This function calculates the least square error of a certain
    model, given a certain dataset. """
    real_densities = df['T Density (veh/km/lane)']
    real_speed = df['T Speed (km/h)']
    pred_speed = model['function'](real_densities, *parameters)
    return np.sqrt(np.sum((real_speed - pred_speed)**2)/len(real_speed))

def fit_data(data, model):
    """ This function fits a certain model to a given dataset by minimizing
    the least squares error. """

    scenarios = set(data.index.get_level_values(2))
    for scenario in scenarios:

        # don't consider intervals without traffic
        df = data.loc[:,:,scenario]
        df = df[df['T Flow (veh/h/lane)'] != 0]

        # optimise
        result = minimize(fun=least_squares_error,
                          x0=model['initial parameters'],
                          args=(df, model), method = 'Powell')

        if not result['success']:
            raise Exception('Could not fit "'+ model['name'] +'" to the data')

        model['optimal parameters ('+scenario.lower()+')'] = result['x']
        model['result of the fitting process ('+scenario.lower()+')'] = result
    return model
