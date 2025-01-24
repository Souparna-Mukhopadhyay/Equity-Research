import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
idx = pd.IndexSlice
import plotly.express as px


# Find the cluster for the selected ticker
def get_cluster_for_symbol(symbol, clusters):
    for cluster in clusters.values():
        if symbol in cluster:
            return cluster
    return []


def test_for_cluster(data_, cluster, symbol, start_date, end_date):
    
    cluster_data = {}
    for ticker in cluster:
        data = data_.loc[:, idx[:, ticker]]
        data.columns = data.columns.droplevel(1)
        if not data.empty:
            data['Day Return'] = (data['Close'] - data['Open']) * 100 / data['Open']
            cluster_data[ticker] = data
    
    if not cluster_data:
        # pass
        print("No data found for the corresponding cluster.")
        return None, None, None, None
    else:
        # Combine data to calculate cluster mean return
        cluster_mean_returns = pd.DataFrame()
        for ticker, data in cluster_data.items():
            cluster_mean_returns[ticker] = data['Day Return']

        cluster_mean_returns['Cluster Mean Return'] = cluster_mean_returns.mean(axis=1)
        # print(cluster_mean_returns)
        # Get the data for the selected symbol
        symbol_data = data_.loc[:, idx[:, symbol]]
        symbol_data.columns = symbol_data.columns.droplevel(1)
        symbol_data['Previous High'] = symbol_data['High'].shift(1)
        # symbol_data['Two Days Ago High'] = symbol_data['High'].shift(2)
        # symbol_data['Condition(B)'] = symbol_data['Previous High'] > symbol_data['Two Days Ago High']

        # Merge with cluster mean returns
        symbol_data = symbol_data.merge(cluster_mean_returns[['Cluster Mean Return']], left_index=True, right_index=True)

        # Filter data where cluster mean return is positive and condition is met
        symbol_data['Previous Day Cluster Mean'] = symbol_data['Cluster Mean Return'].shift(1)
        symbol_data['Cluster Positive(C)'] = symbol_data['Previous Day Cluster Mean'] > 0
        # filtered_data = symbol_data[symbol_data['Condition(B)'] & symbol_data['Cluster Positive(C)']]
        filtered_data = symbol_data[symbol_data['Cluster Positive(C)']]
        

        # Calculate probability of higher high
        filtered_data['Higher High(A)'] = filtered_data['High'] > filtered_data['Previous High']
        # print(filtered_data)
        total_samples = len(filtered_data)
        higher_high_count = filtered_data['Higher High(A)'].sum()
        p_ = (higher_high_count / total_samples) if total_samples > 0 else 0

        # Display results
        # st.write(f"### Results for {symbol}")
        # st.write(f"#### Higher High Probability")
        # st.write(f"Occurance of (C): {total_samples}")
        # st.write(f"Occurance of (A intersect C): {higher_high_count}")
        # st.write(f"P(A/C): {probability_higher_high:.2f}%")
        # p_ = probability_higher_high/100
        sigma_= np.sqrt(p_*(1-p_)/total_samples)
        lower = (p_-1.96*sigma_)*100
        upper = (p_+1.96*sigma_)*100
        z_val = (p_-0.5)/np.sqrt(.25/total_samples)
        p_val_ = 1-norm.cdf(z_val)
        
        # st.write(f"95% confidence interval estimation of probablity lies between {(p_-1.96*sigma_)*100:.2f}% - {(p_+1.96*sigma_)*100:.2f}%")
        return total_samples, higher_high_count, lower, upper, p_val_
            



st.sidebar.header("Instructions")
st.sidebar.write("""
1. This work is intended to answer questions with the aid of statistical hypothesis testing.\n
2. Select the daily open, low, high, close data by selecting the asset symbol and date range for the test.
""")
st.sidebar.header("Configuration")
nifty50_tickers = [
    'ABB.NS', 'ACC.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS',
    'AFFLE.NS', 'AJANTPHARM.NS', 'ALKEM.NS', 'AMBUJACEM.NS', 'APLAPOLLO.NS',
    'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTRAL.NS', 'AUROPHARMA.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
    'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BATAINDIA.NS', 'BERGEPAINT.NS',
    'BHARATFORG.NS', 'BHARTIARTL.NS', 'BHEL.NS', 'BIOCON.NS', 'BOSCHLTD.NS', 'BPCL.NS', 'BRITANNIA.NS', 'BSOFT.NS', 'CANBK.NS',
    'CANFINHOME.NS', 'CAPLIPOINT.NS', 'CASTROLIND.NS', 'CDSL.NS', 'CENTRALBK.NS',  'CIPLA.NS', 'COALINDIA.NS',
    'COFORGE.NS', 'COLPAL.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CROMPTON.NS', 'CUB.NS', 'CUMMINSIND.NS', 'DABUR.NS', 'DALBHARAT.NS',
    'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DELTACORP.NS', 'DIVISLAB.NS', 'DIXON.NS', 'DLF.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'EMAMILTD.NS',
    'ESCORTS.NS', 'EXIDEIND.NS', 'FEDERALBNK.NS', 'FINCABLES.NS', 'FORTIS.NS', 'GAIL.NS', 'GLAND.NS', 'GLENMARK.NS', 'GMRINFRA.NS',
    'GNFC.NS', 'GODREJAGRO.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GRANULES.NS', 'GRASIM.NS', 'GSPL.NS', 'GUJGASLTD.NS', 'HAVELLS.NS',
    'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS',
    'HINDUNILVR.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'IDEA.NS', 'IDFCFIRSTB.NS', 'IEX.NS', 'IGL.NS',
    'INDHOTEL.NS', 'INDIAMART.NS', 'INDIGO.NS', 'INDUSINDBK.NS', 'INDUSTOWER.NS', 'INFY.NS', 'INTELLECT.NS', 'IOC.NS', 'IRCTC.NS',
    'ITC.NS', 'JINDALSTEL.NS', 'JSWENERGY.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'KOTAKBANK.NS', 'LALPATHLAB.NS', 'LAURUSLABS.NS',
    'LICHSGFIN.NS', 'LT.NS', 'LTIM.NS', 'LTTS.NS', 'LUPIN.NS', 'M&M.NS', 'M&MFIN.NS', 'MANAPPURAM.NS', 'MARICO.NS', 'MARUTI.NS',
    'MCX.NS', 'METROPOLIS.NS', 'MFSL.NS', 'MGL.NS', 'MOTHERSON.NS', 'MPHASIS.NS', 'MRF.NS', 'MUTHOOTFIN.NS',
    'NAM-INDIA.NS', 'NBCC.NS', 'NESTLEIND.NS', 'NMDC.NS', 'NTPC.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'PAGEIND.NS', 'PAYTM.NS', 'PEL.NS',
    'PETRONET.NS', 'PFC.NS', 'PFIZER.NS', 'PIIND.NS', 'PNB.NS', 'POLYCAB.NS', 'POWERGRID.NS', 'PVRINOX.NS', 'RAIN.NS', 'RBLBANK.NS',
    'RECLTD.NS', 'RELIANCE.NS', 'SAIL.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SRF.NS',
    'STAR.NS', 'SUNPHARMA.NS', 'SUNTV.NS', 'SYNGENE.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATACONSUM.NS', 'TATAELXSI.NS', 'TATAMOTORS.NS',
    'TATAPOWER.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'THERMAX.NS', 'TIINDIA.NS', 'TORNTPHARM.NS', 'TORNTPOWER.NS', 'TRENT.NS',
    'TVSMOTOR.NS', 'UBL.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'UPL.NS', 'VBL.NS', 'VEDL.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS', 'WIPRO.NS',
    'YESBANK.NS', 'ZEEL.NS', 'ZOMATO.NS'
]

seg = {
    'agrochem': ['PIIND.NS', 'DEEPAKNTR.NS', 'UPL.NS', 'COROMANDEL.NS', 'TATACHEM.NS', 'GNFC.NS', 'SRF.NS'],
    'tyre':['APOLLOTYRE.NS', 'MRF.NS'],
    'heavy-electric':['ABB.NS', 'SIEMENS.NS','CUMMINSIND.NS'],
    'telecom':['INDUSTOWER.NS','IDEA.NS'],
    'bajaj':['BAJAJFINSV.NS', 'BAJFINANCE.NS'],
    'entertainment':['SUNTV.NS','ZEEL.NS'],
    'reality':['DLF.NS','OBEROIRLTY.NS'],
    'asset-management':['HDFCAMC.NS', 'NAM-INDIA.NS'],
    'power-finance':['PFC.NS', 'RECLTD.NS'],
    'psu-bank':['PNB.NS','BANKBARODA.NS', 'CANBK.NS', 'SBIN.NS', 'FEDERALBNK.NS','INDUSINDBK.NS'],
    'banks':['UNIONBANK.NS', 'CENTRALBK.NS', 'RBLBANK.NS', 'CUB.NS', 'YESBANK.NS'],
    'metal_1':['SAIL.NS', 'NMDC.NS', 'HINDCOPPER.NS'],
    'metal_2':['JSWSTEEL.NS', 'HINDALCO.NS', 'TATASTEEL.NS', 'JINDALSTEL.NS', 'APLAPOLLO.NS'],
    'cement':['SHREECEM.NS', 'ULTRACEMCO.NS', 'GRASIM.NS'],
    'oil_1':['IOC.NS','BPCL.NS','HINDPETRO.NS'],
    'energy':['NTPC.NS', 'POWERGRID.NS'],
    'power':['TATAPOWER.NS', 'TORNTPOWER.NS', 'JSWENERGY.NS'],
    'oil_2':['ONGC.NS','GAIL.NS', 'PETRONET.NS'],
    'gas':['GSPL.NS','GUJGASLTD.NS', 'IGL.NS', 'MGL.NS'],
    'adani':['ACC.NS', 'AMBUJACEM.NS', 'ADANIPORTS.NS', 'ADANIENT.NS', 'ADANIPOWER.NS', 'ADANIGREEN.NS'],
    'life':['ICICIGI.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'HDFCLIFE.NS', 'MFSL.NS'],
    'private-bank':['AXISBANK.NS', 'ICICIBANK.NS', 'HDFCBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'INDUSINDBK.NS'],
    'pharma':['SYNGENE.NS','BIOCON.NS', 'GRANULES.NS', 'LAURUSLABS.NS', 'CAPLIPOINT.NS', 'GLAND.NS', 'DIVISLAB.NS',
        'METROPOLIS.NS', 'LALPATHLAB.NS', 'GLENMARK.NS', 'AUROPHARMA.NS', 'APOLLOHOSP.NS', 'AJANTPHARM.NS','ALKEM.NS',
        'LUPIN.NS', 'SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'TORNTPHARM.NS'],
    'auto_1':['ESCORTS.NS', 'MARUTI.NS', 'EICHERMOT.NS','ASHOKLEY.NS', 'TATAMOTORS.NS'],
    'auto_2':['HEROMOTOCO.NS','BAJAJ-AUTO.NS', 'TVSMOTOR.NS', 'M&M.NS'],
    'consumer-goods_1':['ITC.NS','NESTLEIND.NS','BRITANNIA.NS','TATACONSUM.NS','HINDUNILVR.NS','DABUR.NS','GODREJCP.NS','MARICO.NS'],
    # 'consumer-goods_1':['ITC.NS','NESTLEIND.NS','BRITANNIA.NS','TATACONSUM.NS','DABUR.NS','GODREJCP.NS','MARICO.NS'],
    'consumer-goods_2':['COLPAL.NS', 'UBL.NS'],
    'it':['WIPRO.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'MPHASIS.NS',
        'LTTS.NS', 'BSOFT.NS', 'INTELLECT.NS', 'TATAELXSI.NS', 'INFY.NS', 'TCS.NS', 'HCLTECH.NS'],
    'paints':['ASIANPAINT.NS','BERGEPAINT.NS'],
    'us-pharma':['PFIZER.NS','STAR.NS'],
    'electric':['CROMPTON.NS', 'DIXON.NS'],
    'finance_1':['MANAPPURAM.NS', 'MUTHOOTFIN.NS'],
    'finance_2':['M&MFIN.NS', 'SBICARD.NS','IDFCFIRSTB.NS'],
    'house_finance':['LICHSGFIN.NS', 'CANFINHOME.NS'],
    'aditya':['ABCAPITAL.NS', 'ABFRL.NS'],
    'trading':['CDSL.NS', 'MCX.NS'],
    'petrochem':['CASTROLIND.NS','RAIN.NS'],
    'inter-sec-2':['JUBLFOOD.NS', 'VOLTAS.NS'],
    'godrej':['GODREJAGRO.NS','GODREJIND.NS'],
}


default_end_date = datetime.now()
default_start_date = default_end_date - timedelta(days=1460)

with st.sidebar.form("test"):
    
# Sidebar inputs
    symbol = st.selectbox("Select a Ticker", nifty50_tickers)
    start_date = st.date_input("Start Date", value=default_start_date)
    end_date = st.date_input("End Date", value=default_end_date)
    test = st.form_submit_button("Test Hypohesis")

if not test:
    st.write("##### Q1:  Is there any tendency in market to get a higher high given the high of previous day is higher than two days ago high ?\n \n ")
    # st.write("###### Let the event - high of previous day is higher than two days ago high be denoted by B")
    # st.write("###### Let the event - Higher high be denoted by A")
    # st.write("###### Let the event - Higher high and not lower low be denoted by Y")
    # st.write("###### H0: P(A/B) = 0.5 (50%)")
    # st.write("###### H1: P(A/B) > 0.5 (50%)")
    st.write(f"##### Q2:  Knowing that the previous day sectorial mean (Mean of the returns of assets in the corresponding cluster) gain was positive do imply higher chance of higher high ?")
    st.write(f"")
    # st.write(f"###### The symbol {symbol} belongs to the sectorial cluster consists of : \n {selected_cluster}")
    # st.write("###### Let the event - previous day sectorial mean return positive be denoted by C")
    # st.write("###### H0: P(A/C) = 0.5 (50%)")
    # st.write("###### H1: P(A/C) > 0.5 (50%)")

else:
    try:
        st.write("##### Q1: Is there any tendency in market to get a higher high given the high of previous day is higher than two days ago high ?")
        st.write("###### Let the event - high of previous day is higher than two days ago high be denoted by B")
        st.write("###### Let the event - Higher high be denoted by A")
        # st.write("###### Let the event - Higher high and not lower low be denoted by Y")
        st.write("###### H0: P(A/B) = 0.5 (50%)")
        st.write("###### H1: P(A/B) > 0.5 (50%)")
        # st.subheader("Test result")
        
        # Map time period to interval
        # idx = pd.IndexSlice
        data_ = yf.download(nifty50_tickers, start=start_date, end=end_date)
        data = data_.loc[:, idx[:, symbol]]
        data.columns = data.columns.droplevel(1)
        if data.empty:
            st.error("No data found for the given symbol and date range.")
        else:
            # Calculate higher high probabilities
            data['Previous High'] = data['High'].shift(1)
            data['Previous Low'] = data['Low'].shift(1)
            data['Two Days Ago High'] = data['High'].shift(2)
            data['Condition(B)'] = data['Previous High'] > data['Two Days Ago High']
            data['Higher High(A)'] = data['High'] > data['Previous High']
            # data['Higher High and not Lower Low(Y)'] = (data['High'] > data['Previous High']) and (data['Low'] > data['Previous Low'])

            # Calculate lower low probabilities
            # data['Previous Low'] = data['Low'].shift(1)
            # data['Two Days Ago Low'] = data['Low'].shift(2)
            # data['Condition Low'] = data['Previous Low'] < data['Two Days Ago Low']
            # data['Lower Low'] = data['Low'] < data['Previous Low']

            # Filter data for conditions
            condition_high_met = data[data['Condition(B)']]
            # condition_low_met = data[data['Condition Low']]

            # Higher High calculations
            total_samples_high = len(condition_high_met)
            higher_high_count = condition_high_met['Higher High(A)'].sum()
            probability_higher_high = (higher_high_count / total_samples_high * 100) if total_samples_high > 0 else 0

            # if probability_higher_high>=50:
            #     st.write("###### H1: P(A/B) > 0.5 (50%)")
            # else:
            #     st.write("###### H1: P(A/B) < 0.5 (50%)")
            
            st.subheader("Test result")
            
            # # Lower Low calculations
            # total_samples_low = len(condition_low_met)
            # lower_low_count = condition_low_met['Lower Low'].sum()
            # probability_lower_low = (lower_low_count / total_samples_low * 100) if total_samples_low > 0 else 0

            # Display results
            # st.write(f"### Results for {symbol}")

            # Higher High results
            # st.write(f"#### Higher High Probability")
            st.write(f"Occurance of (B) : {total_samples_high}")
            st.write(f"Occurance of (A intersection B): {higher_high_count}")
            st.write(f"P(A/B): {probability_higher_high:.2f}%")
            prob = (probability_higher_high/100)
            z_val = (prob-0.5)/np.sqrt(.25/total_samples_high)
            sigma = np.sqrt(prob*(1-prob)/total_samples_high)
            st.write(f"95% confidence interval estimatiopn of probablity lies between {(prob-1.96*sigma)*100:.2f}% - {(prob+1.96*sigma)*100:.2f}%")
            
            # if probability_higher_high>=50:
            #     p_val = 1-norm.cdf(z_val)
            # else:
            #     p_val = norm.cdf(z_val)
            
            p_val = 1-norm.cdf(z_val)
            
            st.write(f"P value:{p_val:.4f}")
            if p_val>=0.05:
                st.write("Failed to reject H0 with 5% significance. P(A/B) = 50%")
            else:
                st.write("H0 rejected with 5% significance. P(A/B) > 50%")
                
            data['Strategy return percentage'] = (data['Previous High']- data['Open'])*100/data['Open']
            data['Day return percentage'] = (data['Close']- data['Open'])*100/data['Open']
            
            # st.write(f"Expectation : {condition_high_met['Day return percentage'].mean()}")
                
            

            # # Lower Low results
            # st.write(f"#### Lower Low Probability")
            # st.write(f"Total Samples (Condition Met): {total_samples_low}")
            # st.write(f"Lower Low Count: {lower_low_count}")
            # st.write(f"Probability of Lower Low: {probability_lower_low:.2f}%")

            # Show data table
            st.write("#### Data Table")
            # st.dataframe(data[['High', 'Previous High', 'Two Days Ago High', 'Condition High', 'Higher High',  'Low',
            #                 'Previous Low', 'Two Days Ago Low', 'Condition Low', 'Lower Low']])
            st.dataframe(data[['High', 'Previous High', 'Two Days Ago High', 'Condition(B)', 'Higher High(A)', 'Strategy return percentage', 'Day return percentage']])

            selected_cluster = get_cluster_for_symbol(symbol, seg)

            if not selected_cluster:
                pass
                st.write(f"##### Q2: Knowing that the previous day sectorial mean (Mean of the returns of assets in the corresponding cluster) gain was positive do imply higher chance of higher high ?")
                # st.write(f"###### The symbol {symbol} belongs to the sectorial cluster consists of : \n {selected_cluster}")
                st.error(f"The symbol {symbol} does not belong to any predefined sectorial cluster.")
            else:
                st.write(f"##### Q2: Knowing that the previous day sectorial mean (Mean of the returns of assets in the corresponding cluster) gain was positive do imply higher chance of higher high ?")
                st.write(f"###### The symbol {symbol} belongs to the sectorial cluster consists of : \n {selected_cluster}")

                st.write("###### Let the event - previous day sectorial mean return positive be denoted by C")
                st.write("###### H0: P(A/C) = 0.5 (50%)")
                st.write("###### H1: P(A/C) > 0.5 (50%)")
                # total_samples, higher_high_count, _, _ = p_val(selected_cluster, symbol, start_date, end_date)
                cluster_data = {}
                for ticker in selected_cluster:
                    # data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
                    data = data_.loc[:, idx[:, ticker]]
                    data.columns = data.columns.droplevel(1)
                    if not data.empty:
                        data['Day Return'] = (data['Close'] - data['Open']) * 100 / data['Open']
                        cluster_data[ticker] = data

                if not cluster_data:
                    st.error("No data found for the corresponding cluster.")
                else:
                    # Combine data to calculate cluster mean return
                    cluster_mean_returns = pd.DataFrame()
                    for ticker, data in cluster_data.items():
                        cluster_mean_returns[ticker] = data['Day Return']

                    cluster_mean_returns['Cluster Mean Return'] = cluster_mean_returns.mean(axis=1)

                    # Get the data for the selected symbol
                    symbol_data = cluster_data[symbol]
                    symbol_data['Previous High'] = symbol_data['High'].shift(1)
                    symbol_data['Two Days Ago High'] = symbol_data['High'].shift(2)
                    symbol_data['Condition(B)'] = symbol_data['Previous High'] > symbol_data['Two Days Ago High']
                    symbol_data['day return percentage'] = (symbol_data['Close']- symbol_data['Open'])*100/symbol_data['Open']
                    symbol_data['strategy return percentage'] = (symbol_data['Previous High']- symbol_data['Open'])*100/symbol_data['Open']
                    # Merge with cluster mean returns
                    symbol_data = symbol_data.merge(cluster_mean_returns[['Cluster Mean Return']], left_index=True, right_index=True)

                    # Filter data where cluster mean return is positive and condition is met
                    symbol_data['Previous Day Cluster Mean'] = symbol_data['Cluster Mean Return'].shift(1)
                    symbol_data['Cluster Positive(C)'] = symbol_data['Previous Day Cluster Mean'] > 0
                    symbol_data['Higher High(A)'] = symbol_data['High'] > symbol_data['Previous High']
                    # filtered_data = symbol_data[symbol_data['Condition(B)'] & symbol_data['Cluster Positive(C)']]
                    filtered_data = symbol_data[symbol_data['Cluster Positive(C)']]
                    

                    # Calculate probability of higher high
                    
                    total_samples = len(filtered_data)
                    higher_high_count = filtered_data['Higher High(A)'].sum()
                # if total_samples != None:
                    probability_higher_high = (higher_high_count / total_samples * 100) if total_samples > 0 else 0

                    # Display results
                    # st.write(f"### Results for {symbol}")
                    # st.write(f"#### Higher High Probability")
                    st.write(f"Occurance of (C): {total_samples}")
                    st.write(f"Occurance of (A intersect C): {higher_high_count}")
                    st.write(f"P(A/C): {probability_higher_high:.2f}%")
                    p_ = probability_higher_high/100
                    sigma_= np.sqrt(p_*(1-p_)/total_samples)
                    st.write(f"95% confidence interval estimation of probablity lies between {(p_-1.96*sigma_)*100:.2f}% - {(p_+1.96*sigma_)*100:.2f}%")
                    if p_-1.96*sigma_<=0.5:
                        st.write("Failed to reject H0 and conclude that no signifiacnt tendency for A given C.")
                    else:
                        st.write("Rejected H0 and concluded to have significant tendency of A given C.")
                    # Show data table
                    st.write("### Data Table")
                    st.dataframe(symbol_data[['High', 'Previous High', 'Two Days Ago High', 'Condition(B)',
                                                'Previous Day Cluster Mean', 'Cluster Positive(C)','Open', 'Higher High(A)', 'strategy return percentage', 'day return percentage']])
                    
                    # filtered_data = filtered_data[filtered_data['Open']< filtered_data['Previous High']]
                    # print(len(filtered_data))
                    # filtered_data["Day return"] = np.where(
                    #     filtered_data["High"] > filtered_data["Previous High"],
                    #     (filtered_data["Previous High"] - filtered_data["Open"]) * 100 / filtered_data["Open"],
                    #     (filtered_data["Close"] - filtered_data["Open"]) * 100 / filtered_data["Open"])

                    # # filtered_data["Day return"] = (filtered_data["Previous High"] - filtered_data["Open"])*100/filtered_data["Open"] if (filtered_data["High"]>filtered_data["Previous High"]) else (filtered_data["Close"]- filtered_data["Open"])*100/filtered_data["Open"]
                    # # st.write(f"### Day return histogram when Event C occurs")
                    # mean_ = filtered_data["Day return"].mean()
                    # median_ = filtered_data["Day return"].median()
                    # var_ = filtered_data["Day return"].var()
                    # plt.figure(figsize=(10, 6))
                    # sns.histplot(filtered_data['Day return'], kde=True, stat="density", bins=30, color='blue', label='Returns')
                    # plt.axvline(mean_, color='r', linestyle='--', label=f'Mean: {mean_:.2f}')
                    # plt.axvline(median_, color='g', linestyle='-', label=f'Median: {median_:.2f}')
                    # # Draw the vertical line at the calculated quantile
                    # # plt.axvline(uslx, color='black', linestyle=':', label=f'USL = {uslx:.2f}')
                    # # plt.axvline(lslx, color='black', linestyle=':', label=f'LSL = {lslx:.2f}')
                    # plt.title(f'Day return histogram when Event C occurs for {symbol}')
                    # plt.legend(title=f"Variance: {var_:.2f}")
                    # st.pyplot(plt)
                    
            st.write("#### Confidence interval estimation of probablity of event A given a sector mean is positive.")
            hypothesis_test_table = []
            # data_ = yf.download(symbol, start=start_date, end=end_date, interval='1d')
            for sec_name, tickers in seg.items():
                # print("yes")
                total_samples, higher_high_count, lower, upper, p_value_ = test_for_cluster(data_, tickers, symbol, start_date, end_date)
                # print("ok")
                test_table = {}
                test_table['Sector'] = sec_name
                test_table['Occurance of C'] = total_samples
                test_table['Occurance of C intersect A'] = higher_high_count
                test_table['Lower limit(%)'] = lower
                test_table['Upper limit(%)'] = upper
                # test_table['P-value(%)'] = p_value_*100
                hypothesis_test_table.append(test_table)
            hypothesis_test_dataframe = pd.DataFrame(hypothesis_test_table)
            st.dataframe(hypothesis_test_dataframe)
            
            
            # Streamlit app
            # st.title("Confidence Intervals for Different Sectors")

            # st.write(
            #     "This visualization shows the probability confidence intervals (in percentages) for various sectors. "
            #     "The lower and upper limits are displayed as ranges."
            # )

            # Create a new column for the range of confidence intervals
            fig = px.scatter(
                hypothesis_test_dataframe,
                x="Sector",
                y=0.5 * (hypothesis_test_dataframe["Upper limit(%)"] + hypothesis_test_dataframe["Lower limit(%)"]),
                error_y= 0.5 * (hypothesis_test_dataframe["Upper limit(%)"] - hypothesis_test_dataframe["Lower limit(%)"]),
                labels={
                    "Lower limit(%)": "Confidence Interval (%)",
                    "Sector": "Sector",
                },
                title="Confidence Intervals by Sector",
            )

            # Customize the plot
            fig.update_traces(marker=dict(size=8), line=dict(width=2))
            fig.update_layout(
                yaxis_title="Percentage (%)",
                xaxis_title="Sector",
                showlegend=False,
            )

            st.plotly_chart(fig)
              
    except Exception as e:
        st.error(f"An error occurred: {e}")