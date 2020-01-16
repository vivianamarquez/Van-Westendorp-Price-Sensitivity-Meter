# Import Libraries

import numpy as np
import pandas as pd
from functools import reduce

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import warnings
warnings.filterwarnings("ignore")


# Remove intransitive price preferences
def validate(df, price_cols):
    '''
    Input: A data frame and a list with the price column names
    Output: A data frame with only valid responses
    '''
    original_size = df.shape[0]
    validations = []
    for val1, val2 in zip(price_cols, price_cols[1:]):
        validations.append(df[val1] < df[val2])
    df = df[validations[0] & validations[1] & validations[2]]
    new_size = df.shape[0]
    print (f"Total data set contains {original_size} cases, {new_size} cases were kept (transitive price preferences).\n")
    return df

# Compute cumulative frequencies
def cdf(df, col):
    '''
    Input: A data frame and a column for which we wish to obtain its CDF
    Output: A pd.series object with the column's CDF
    '''
    # Frequency
    stats_df = df.groupby(col)[col].agg('count').pipe(pd.DataFrame).rename(columns = {col: f'{col}_frequency'})

    # PDF
    stats_df[f'{col}_pdf'] = stats_df[f'{col}_frequency'] / sum(stats_df[f'{col}_frequency'])

    # CDF
    stats_df[f'{col}_cdf'] = stats_df[f'{col}_pdf'].cumsum()
    stats_df.reset_index(inplace=True)
    stats_df.drop([f'{col}_frequency', f'{col}_pdf'], axis=1, inplace=True)
    stats_df.rename(columns = {col: 'Price', f'{col}_cdf': col}, inplace=True)
    
    return stats_df

def cdf_table(df, price_cols, interpolate=True):
    '''
    Re-creating R's function output$data_vanwestendorp
    '''
    df.rename(columns={price_cols[0]: "Too Cheap", price_cols[1]: "Cheap", price_cols[2]: "Expensive", price_cols[3]: "Too Expensive"})
    cdfs = [cdf(df, 'Too Cheap'), cdf(df, 'Cheap'), cdf(df, 'Expensive'), cdf(df, 'Too Expensive')]
    cdfs = reduce(lambda left, right: pd.merge(left, right, on=['Price'], how='outer'), cdfs).sort_values('Price')
    cdfs = cdfs.fillna(method='ffill').fillna(0)
    cdfs['Too Cheap'] = 1 - cdfs['Too Cheap']
    cdfs['Cheap'] = 1 - cdfs['Cheap']
    cdfs['Not Cheap'] = 1 - cdfs['Cheap']
    cdfs['Not Expensive'] = 1 - cdfs['Expensive']
    cdfs = cdfs.clip(lower=0)
    if interpolate == True:
        low = cdfs.Price.min()
        high = cdfs.Price.max()
        cdfs = pd.merge(pd.DataFrame(list(np.arange(low,high,0.01)), columns = ['Price']), cdfs, how='outer').sort_values('Price')
        cdfs['Price'] = cdfs['Price'].apply(lambda value: round(float(value),2))
        cdfs.drop_duplicates(['Price'], keep='last', inplace=True)
        cdfs = cdfs.interpolate(method ='linear', limit_direction ='forward')
        cdfs['Too Cheap'] = cdfs['Too Cheap'].fillna(1)
        cdfs['Cheap'] = cdfs['Cheap'].fillna(0)
        cdfs['Expensive'] = cdfs['Expensive'].fillna(0)
        cdfs['Too Expensive'] = cdfs['Too Expensive'].fillna(0)
        cdfs['Not Cheap'] = cdfs['Not Cheap'].fillna(0)
        cdfs['Not Expensive'] = cdfs['Not Expensive'].fillna(1)
        cdfs.reset_index(inplace=True)
        cdfs.drop('index', axis=1, inplace=True)
    return cdfs    


# Plot function
def plot_function(cdfs, 
                  Point_of_Marginal_Cheapness, PMC_height,
                  Point_of_Marginal_Expensiveness, PME_height,
                  Indifference_Price_Point, IPP_height,
                  Optimal_Price_Point, OPP_height,
                  title=""):
    line_width = 1
    marker_size = 3

    var = "Too Expensive"
    trace1 = go.Scatter(
                    x=cdfs.Price.values,
                    y=cdfs[var].values,
                    text=[f"{var}<br>Price: ${price:.2f}<br>Participants: {val*100:.2f}%" for (price,val) in zip(cdfs.Price.values,cdfs[var].values)],
                        mode='lines', 
                        opacity=0.8,
                        marker={
                            'size': marker_size,
                            'color': "red"
                        },
                        hoverinfo='text',
                        line = {
                            'color': "red",
                            'width':line_width
                        },
                        name=var
                    ) 

    var = "Not Expensive"
    trace2 = go.Scatter(
                    x=cdfs.Price.values,
                    y=cdfs[var].values,
                    text=[f"{var}<br>Price: ${price:.2f}<br>Participants: {val*100:.2f}%" for (price,val) in zip(cdfs.Price.values,cdfs[var].values)],
                        mode='lines', 
                        opacity=0.8,
                        marker={
                            'size': marker_size,
                            'color': "orange"
                        },
                        hoverinfo='text',
                        line = {
                            'color': "orange",
                            'width':line_width
                        },
                        name=var
                    ) 

    var = "Not Cheap"
    trace3 = go.Scatter(
                    x=cdfs.Price.values,
                    y=cdfs[var].values,
                    text=[f"{var}<br>Price: ${price:.2f}<br>Participants: {val*100:.2f}%" for (price,val) in zip(cdfs.Price.values,cdfs[var].values)],
                        mode='lines', 
                        opacity=0.8,
                        marker={
                            'size': marker_size,
                            'color': "blue"
                        },
                        hoverinfo='text',
                        line = {
                            'color': "blue",
                            'width': line_width
                        },
                        name=var
                    ) 

    var = "Too Cheap"
    trace4 = go.Scatter(
                    x=cdfs.Price.values,
                    y=cdfs[var].values,
                    text=[f"{var}<br>Price: ${price:.2f}<br>Participants: {val*100:.2f}%" for (price,val) in zip(cdfs.Price.values,cdfs[var].values)],
                        mode='lines', 
                        opacity=0.8,
                        marker={
                            'size': marker_size,
                            'color': "green"
                        },
                        hoverinfo='text',
                        line = {
                            'color': "green",
                            'width':line_width
                        },
                        name=var
                    ) 

    point1 = go.Scatter(
                    x=[Point_of_Marginal_Cheapness],
                    y=[PMC_height],
                    text=[f"Point of Marginal Cheapness: ${Point_of_Marginal_Cheapness:.2f}<br>Participants: {PMC_height*100:.2f}%"],
                        mode='markers', 
                        opacity=1,
                        marker={
                            'size': 7,
                            'color': "blue"
                        },
                        hoverinfo='text',
                        name=f"<br>Point of Marginal Cheapness<br>${Point_of_Marginal_Cheapness:.2f}"
                    ) 

    point2 = go.Scatter(
                    x=[Point_of_Marginal_Expensiveness],
                    y=[PME_height],
                    text=[f"Point of Marginal Expensiveness: ${Point_of_Marginal_Expensiveness:.2f}<br>Participants: {PME_height*100:.2f}%"],
                        mode='markers', 
                        opacity=1,
                        marker={
                            'size': 7,
                            'color': "red"
                        },
                        hoverinfo='text',
                        name=f"Point of Marginal Expensiveness<br>${Point_of_Marginal_Expensiveness:.2f}"
                    ) 

    point3 = go.Scatter(
                    x=[Indifference_Price_Point],
                    y=[IPP_height],
                    text=[f"Indifference Price Point: ${Indifference_Price_Point:.2f}<br>Participants: {IPP_height*100:.2f}%"],
                        mode='markers', 
                        opacity=1,
                        marker={
                            'size': 7,
                            'color': "orange"
                        },
                        hoverinfo='text',
                        name=f"Indifference Price Point<br>${Indifference_Price_Point:.2f}"
                    ) 

    point4 = go.Scatter(
                    x=[Optimal_Price_Point],
                    y=[OPP_height],
                    text=[f"Optimal Price Point: ${Optimal_Price_Point:.2f}<br>Participants: {OPP_height*100:.2f}%"],
                        mode='markers', 
                        opacity=1,
                        marker={
                            'size': 7,
                            'color': "green"
                        },
                        hoverinfo='text',
                        name=f"Optimal Price Point<br>${Optimal_Price_Point:.2f}"
                    ) 

    data = [trace1, trace2, trace3, trace4, point1, point2, point3, point4]

    layout = go.Layout(title=f"Van Westendorp's Price Sensitivity Meter<br>{title}",
                    xaxis=dict(title='$ Price', range=(cdfs.Price.min()-5, cdfs.Price.max()+5)),
                    yaxis=dict(title='% of Participants', range=(-0.1,1.1)),
                    template="plotly_white"
                    )
    
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

    return None


# Get results
def results(df, price_cols, plot=True, plot_title=""):
    df = validate(df, price_cols)
    cdfs = cdf_table(df, price_cols)

    Point_of_Marginal_Cheapness = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Too Cheap'] - cdfs['Not Cheap']))).flatten()+1]['Price'].values[0]
    Point_of_Marginal_Expensiveness = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Too Expensive'] - cdfs['Not Expensive']))).flatten()+1]['Price'].values[0]
    Indifference_Price_Point = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Not Cheap'] - cdfs['Not Expensive']))).flatten()+1]['Price'].values[0]
    Optimal_Price_Point = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Too Expensive'] - cdfs['Too Cheap']))).flatten()+1]['Price'].values[0]

    # For the plot
    PMC_height = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Too Cheap'] - cdfs['Not Cheap']))).flatten()+1][['Too Cheap', 'Not Cheap']].mean(axis=1).values[0]
    PME_height = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Too Expensive'] - cdfs['Not Expensive']))).flatten()+1][['Too Expensive', 'Not Expensive']].mean(axis=1).values[0]
    IPP_height = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Not Cheap'] - cdfs['Not Expensive']))).flatten()+1][['Not Cheap', 'Not Expensive']].mean(axis=1).values[0]
    OPP_height = cdfs.iloc[np.argwhere(np.diff(np.sign(cdfs['Too Expensive'] - cdfs['Too Cheap']))).flatten()+1][['Too Expensive', 'Too Cheap']].mean(axis=1).values[0]

    print(f"Accepted Price Range: ${Point_of_Marginal_Cheapness:.2f} - ${Point_of_Marginal_Expensiveness:.2f}")
    print(f"Indifference Price Point: ${Indifference_Price_Point:.2f}")
    print(f"Optimal Price Point: ${Optimal_Price_Point:.2f}")

    if plot==True:
        plot_function(cdfs, 
                      Point_of_Marginal_Cheapness, PMC_height,
                      Point_of_Marginal_Expensiveness, PME_height,
                      Indifference_Price_Point, IPP_height,
                      Optimal_Price_Point, OPP_height,
                      plot_title)


'''
If you'd like to contribute to make this code better, write me at Twitter @vivmarquez
If you thought it was useful, also tweet me, it would make me happy :)
'''
