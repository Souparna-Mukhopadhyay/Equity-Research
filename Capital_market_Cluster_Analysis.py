import streamlit as st
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import random
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.figure_factory as ff
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# convert the redundant n*n square matrix form into a condensed nC2 array
    

# Sidebar inputs
stocks = [
    'ABB.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ACC.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS',
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

# 'CENTURYTEX.NS',

st.sidebar.write("### Configuration")
st.write("#### This work is intended to visualize the clusters present in the capital market on the basis of similarity of daily returns of the capital assets.")
st.write("###### 1. First the clusters are visualized with 2 dimensional t-sne plot. For hierarchical clustering, the dendogram structure also presented.")
st.write("###### 2. Then Silhouette Scores for different number of clusters are presented to identify the number of clusters present in the data.")
st.write("###### 3. Finally all clusters and the corresponding assets are presented.")

with st.sidebar.form("Segmentation"):
    start_date = st.date_input("Select start date", datetime(2020, 1, 1))
    end_date = st.date_input("Select end date", datetime(2024, 12, 31))
    analysis_type = st.selectbox("Choose Return Type", ['Daily Day', 'Daily Night', 'Daily'])
    perplexity = st.number_input("Perplexity", min_value=1, max_value=50, value=8)
    state = st.checkbox("Fix random state", value=True)
    max_clusters = st.slider("Number of Clusters", min_value=2, max_value=80, value=50)
    clustering_method = st.selectbox("Select Clustering Method", ['Hierarchical Clustering', 'K-Means'])
    # st.write("Note: Linkage method is valid for only ")
    linkage_method = st.selectbox("Linkage Method", ['single', 'complete', 'ward'])
    cluster_on = st.selectbox("Cluster On", ['t-SNE Points', 'Returns'])
    button = st.form_submit_button("Apply Changes")

if button or "form_submitted" not in st.session_state:
    
    # Download stock data
    data = yf.download(stocks, start=start_date, end=end_date)

    # Calculate returns
    if analysis_type == 'Daily':
        returns = data['Close'].pct_change().dropna()
    elif analysis_type == 'Daily Night':
        returns = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    else:
        returns = (data['Close'] - data['Open']) / data['Open']

    returns.dropna(inplace=True)

    # Correlation matrix
    correlation_matrix = returns.corr()

    # Create a distance matrix (1 - correlation)
    distance_matrix = 1 - correlation_matrix

    # t-SNE transformation
    if state:
        if "random_state" not in st.session_state:
            st.session_state['random_state'] = random.randint(0, 100000)
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=st.session_state['random_state'], perplexity=perplexity)
    else:
        st.session_state['random_state'] = random.randint(0, 100000)
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=st.session_state['random_state'], perplexity=perplexity)
        
    tsne_results = tsne.fit_transform(distance_matrix)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'], index=correlation_matrix.index)
    tsne_df['Ticker'] = tsne_df.index

    # Perform hierarchical clustering
    if clustering_method == "Hierarchical Clustering":
        
        if cluster_on == 'Returns':
            linkage_matrix = linkage(returns.T, method=linkage_method, optimal_ordering=True)
            clustering_data = returns.T
        else:
            linkage_matrix = linkage(tsne_results, method=linkage_method, optimal_ordering=True)
            clustering_data = tsne_df[['TSNE1', 'TSNE2']]

        # Create dendrogram
        fig_dendrogram = ff.create_dendrogram(
            clustering_data,
            orientation='left',
            color_threshold=2.48,
            labels=correlation_matrix.index.tolist(),
            linkagefun=lambda x: linkage_matrix
        )
        fig_dendrogram.update_layout(title='Hierarchical Clustering Dendrogram', height=1000)

        silhouette_scores = []
        # Generate cluster labels
        for cluster_number in range(2, 100):
            cluster_labels = fcluster(linkage_matrix, t=cluster_number, criterion='maxclust')
            clusters = [int(i-1) for i in cluster_labels]
            # print(clusters)
            sil_avg = silhouette_score(clustering_data, clusters)
            silhouette_scores.append(sil_avg)
        cluster_labels = fcluster(linkage_matrix, t=max_clusters, criterion='maxclust')
        print(cluster_labels)
        tsne_df['Cluster'] = cluster_labels
        
        title = f"t-SNE Visualization with {linkage_method.capitalize()} linkage Hierarchical Clustering"
        
    else:
        
        if cluster_on == 'Returns':
            clustering_data = distance_matrix
        else:
            clustering_data = tsne_results
        
        silhouette_scores = []
        for cluster_number in range(2,100):
            
            kmeans = KMeans(n_clusters=cluster_number)
            clusters = kmeans.fit_predict(clustering_data)
            sil_avg = silhouette_score(clustering_data, clusters)
            silhouette_scores.append(sil_avg)
            
        kmeans = KMeans(n_clusters=max_clusters)
        clusters = kmeans.fit_predict(clustering_data)
        print(clusters)
        tsne_df['Cluster'] = clusters

        # # Silhouette score
        # silhouette_avg = silhouette_score(clustering_data, clusters)
        # st.sidebar.write(f"Silhouette Score: {silhouette_avg:.2f}")
        
        title = f"t-SNE Visualization with K-means Clustering"
        
    
    
    # Plot t-SNE scatter with clusters
    fig_tsne_clusters = px.scatter(
        tsne_df, x='TSNE1', y='TSNE2', color=tsne_df['Cluster'].astype(str), text='Ticker',
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    # color=tsne_df['Cluster'].astype(str),
    fig_tsne_clusters.update_traces(textposition='top center')
    fig_tsne_clusters.update_layout(height=800)

    # Display in Streamlit
    # st.plotly_chart(fig_dendrogram)
    st.plotly_chart(fig_tsne_clusters)
    if clustering_method == "Hierarchical Clustering":
        st.plotly_chart(fig_dendrogram)
    
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(2,100)), silhouette_scores, marker='o', linestyle='--')
    plt.title("Silhouette Scores for Different Number of Clusters", fontsize=14)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    # plt.xticks(50)
    plt.grid()
    # plt.show()
    st.pyplot(plt)
    
    st.write("##### Below are the clusters and their associated tickers:")

    # Group by Cluster
    clusters = tsne_df.groupby("Cluster")

    # Display clusters and tickers
    for cluster, group in clusters:
        st.write(f"###### Cluster {cluster}")
        tickers = group["Ticker"].tolist()
        st.write(", ".join(tickers))