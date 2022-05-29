from audioop import avg
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
pd.set_option('display.max_columns',50)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
from sklearn.cluster import KMeans

#load dataset
data = pd.read_csv("D:\microsoft\proj\cars_ds_final.csv.zip")

#clean data i.e remove the redundancies
def clean():
    global data
    data['car'] = data.Make + ' ' + data.Model
    c = ['Make','Model','car','Variant','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain','Ex-Showroom_Price','Displacement','Cylinders',
        'ARAI_Certified_Mileage','Power','Torque','Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase','Number_of_Airbags']
    data_full = data.copy()
    data['Ex-Showroom_Price'] = data['Ex-Showroom_Price'].str.replace('Rs. ','',regex=False)
    data['Ex-Showroom_Price'] = data['Ex-Showroom_Price'].str.replace(',','',regex=False)
    data['Ex-Showroom_Price'] = data['Ex-Showroom_Price'].astype(int)
    data = data[c]
    data = data[~data.ARAI_Certified_Mileage.isnull()]
    data = data[~data.Make.isnull()]
    data = data[~data.Width.isnull()]
    data = data[~data.Cylinders.isnull()]
    data = data[~data.Wheelbase.isnull()]
    data = data[~data['Fuel_Tank_Capacity'].isnull()]
    data = data[~data['Seating_Capacity'].isnull()]
    data = data[~data['Torque'].isnull()]
    data['Height'] = data['Height'].str.replace(' mm','',regex=False).astype(float)
    data['Length'] = data['Length'].str.replace(' mm','',regex=False).astype(float)
    data['Width'] = data['Width'].str.replace(' mm','',regex=False).astype(float)
    data['Wheelbase'] = data['Wheelbase'].str.replace(' mm','',regex=False).astype(float)
    data['Fuel_Tank_Capacity'] = data['Fuel_Tank_Capacity'].str.replace(' litres','',regex=False).astype(float)
    data['Displacement'] = data['Displacement'].str.replace(' cc','',regex=False)
    data.loc[data.ARAI_Certified_Mileage == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
    data.loc[data.ARAI_Certified_Mileage == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
    data['ARAI_Certified_Mileage'] = data['ARAI_Certified_Mileage'].str.replace(' km/litre','',regex=False).astype(float)
    data.Number_of_Airbags.fillna(0,inplace= True)
    data['price'] = data['Ex-Showroom_Price'] * 0.014
    data.drop(columns='Ex-Showroom_Price', inplace= True)
    data.price = data.price.astype(int)
    HP = data.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
    HP = HP.apply(lambda x: round(x,2))
    TQ = data.Torque.str.extract(r'(\d{1,4}).*').astype(int)
    TQ = TQ.apply(lambda x: round(x,2))
    data.Torque = TQ
    data.Power = HP
    data.Doors = data.Doors.astype(int)
    data.Seating_Capacity = data.Seating_Capacity.astype(int)
    data.Number_of_Airbags = data.Number_of_Airbags.astype(int)
    data.Displacement = data.Displacement.astype(int)
    data.Cylinders = data.Cylinders.astype(int)
    data.columns = ['make', 'model','car', 'variant', 'body_type', 'fuel_type', 'fuel_system','type', 'drivetrain', 'displacement', 'cylinders',
                'mileage', 'power', 'torque', 'fuel_tank','height', 'length', 'width', 'doors', 'seats', 'wheelbase','airbags', 'price']



#title
st.title("Data analysis of car dataset")
st.text('Exploring a car dataset')

#to make the navigation bar
selected=option_menu(
    menu_title=None,
    options=["Raw data","Cleaned data","Relationship analysis","Scatter plots"],
    orientation="horizontal",
    icons=['activity','activity','reception-4','graph-up'],

)

#contents of raw data option
if selected=="Raw data":
    st.title("Raw data attributes")
    shape=st.checkbox('Display the shape of data')
    if shape:
        st.write(data.shape)

    st.write('First 5 rows of the dataset:')
    st.write(data.head())

    st.write('Last 5 rows of the dataset:')
    st.write(data.tail())

    disp_desc = st.checkbox('Display description of the dataset')
    if disp_desc:
        st.write(data.describe())

    dataset_stats=st.checkbox('Display dataset statistics')
    if dataset_stats:
        l_D = len(data)
        c_m = len(data.Make.unique())
        c_c = len(data.Model.unique())
        n_f = len(data.columns)
        fig = px.bar(x=['Observations',"Makers",'Models','Features'],y=[l_D,c_m,c_c,n_f], width=800,height=400)
        fig.update_layout(
            title="Dataset Statistics",
            xaxis_title="",
            yaxis_title="Counts",
            font=dict(
                size=16,
            )
        )
        st.write(fig)


    num_of_null_val=st.checkbox('Display number of NULL values in each column')
    if num_of_null_val:
        st.write(data.isnull().sum())


#contents of cleaned data option
if selected=="Cleaned data":
    st.title("Cleaned data attributes")
    clean()

    shape=st.checkbox('Display the shape of data')
    if shape:
        st.write(data.shape)

    disp_cols=st.checkbox('Display names of all columns')
    if disp_cols:
        st.write(data.columns)

    st.write('First 5 rows of the dataset:')
    st.write(data.head())

    st.write('Last 5 rows of the dataset:')
    st.write(data.tail())

    disp_desc = st.checkbox('Display description of the dataset')
    if disp_desc:
        st.write(data.describe())



#relationship analysis
if selected=="Relationship analysis":
    st.title("Relationship analysis")
    clean()

    boxplt=st.checkbox("Box plot of car price")
    if boxplt:
        fig=plt.figure(figsize=(12,6))
        #ax1, ax2 = fig.subplots(2, 1, sharey=True,sharex=True)
        sns.boxplot(data=data, x='price',width=.3,color='blue', hue= 'fuel_type')
        plt.title('Box plot of Price',fontsize=18)
        plt.xticks([i for i in range(0,800000,100000)],[f'{i:,}' for i in range(0,800000,100000)],fontsize=14)
        plt.xlabel('price',fontsize=14);
        st.write(fig)
        


    carprice=st.checkbox("Histogram of car price")
    if carprice:
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,11))
        sns.histplot(data=data, x='price',bins=100, alpha=.4, color='green', ax=ax1)
        ax12 = ax1.twinx()
        sns.kdeplot(data=data, x='price', alpha=.2,fill= True,color="#1f406e",ax=ax12,linewidth=0)
        ax12.grid()
        ax1.set_title('Histogram of car price',fontsize=14)
        ax1.set_xlabel('')
        logbins = np.logspace(np.log10(3000),np.log10(744944.578),50)
        sns.histplot(data=data, x='price',bins=logbins,alpha=.6, color='green',ax=ax2)
        ax2.set_title('Histogram of car price in log scale',fontsize=14)
        ax2.set_xscale('log')
        ax22 = ax2.twinx()
        ax22.grid()
        sns.kdeplot(data=data, x='price', alpha=.2,fill= True,color="#1f406e",ax=ax22,log_scale=True,linewidth=0)
        ax2.set_xlabel('Price(log scale)', fontsize=14)
        ax22.set_xticks((800,1000,10000,100000,1000000))
        ax2.xaxis.set_tick_params(labelsize=12);
        ax1.xaxis.set_tick_params(labelsize=12);
        st.write(fig)


    bodytype=st.checkbox("Bar graph of body types")
    if bodytype:
        fig=plt.figure(figsize=(16,7))
        sns.countplot(data=data, y='body_type',alpha=.6,color='maroon')
        plt.title('By body type',fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('')
        plt.ylabel('');
        st.write(fig)

    price_body=st.checkbox("Box plot of price vs body type")
    if price_body:
        fig=plt.figure(figsize=(12,6))
        sns.boxplot(data=data, x='price', y='body_type', palette='viridis')
        plt.title('Box plot of Price of every body type',fontsize=14)
        plt.ylabel('')
        plt.yticks(fontsize=12)
        plt.xticks([i for i in range(0,800000,100000)],[f'{i:,}' for i in range(0,800000,100000)],fontsize=12);
        st.write(fig)

    
    fuel=st.checkbox("Cars with respect to fuel type")
    if fuel:
        fig=plt.figure(figsize=(11,6))
        sns.countplot(data=data, x='fuel_type',alpha=.6, color='darkblue')
        plt.title('Cars count by engine fuel type',fontsize=18)
        plt.xlabel('Fuel Type', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('');
        st.write(fig)

    eng=st.checkbox("Cars with respect to engine type")
    if eng:
        fig=plt.figure(figsize=(14,6))
        sns.histplot(data=data, x='displacement',alpha=.6, color='darkblue',bins=10)
        plt.title('Cars by engine size (in CC)',fontsize=18)
        plt.xticks(fontsize=13);
        plt.yticks(fontsize=13);
        st.write(fig)

    pow=st.checkbox("Cars with respect to power")
    if pow:
        fig=plt.figure(figsize=(14,6))
        sns.histplot(data=data, x='power',alpha=.6, color='darkgreen')
        plt.title('Cars by power',fontsize=18);
        plt.xticks(fontsize=13);
        plt.yticks(fontsize=13);
        st.write(fig)

    powpri=st.checkbox("Power versus price")
    if powpri:
        st.subheader("Relationship between power and price ")
        fig=plt.figure(figsize=(10,8))
        sns.scatterplot(data=data, x='power', y='price',hue='body_type',palette='viridis',alpha=.89, s=120 );
        plt.xticks(fontsize=12);
        plt.yticks(fontsize=12)
        plt.xlabel('Power',fontsize=14)
        plt.ylabel('Price',fontsize=14)
        plt.title('Relation between power and price',fontsize=20);
        st.write(fig)

    cor_mat=st.checkbox("Show the co-relation matrix")
    if cor_mat:
        graph=plt.figure(figsize=(22,8))
        sns.heatmap(data.corr(), annot=True, fmt='.2%')
        plt.title('Correlation between differet variable',fontsize=20)
        plt.xticks(fontsize=14, rotation=320)
        plt.yticks(fontsize=14);
        st.write(graph)

    pairplt=st.checkbox("Display the pair plots")
    if pairplt:
        graph=sns.pairplot(data,vars=[ 'displacement', 'mileage', 'power', 'price','fuel_tank'], hue= 'fuel_type',
        palette=sns.color_palette('magma',n_colors=4),diag_kind='kde',height=2, aspect=1.8);
        st.pyplot(graph)


        

if selected=="Scatter plots":
    st.title("Scatter plots")
    clean()


    num_cols = [ i for i in data.columns if data[i].dtype != 'object']
    km = KMeans(n_clusters=8, n_init=20, max_iter=400, random_state=0)
    clusters = km.fit_predict(data[num_cols])
    data['cluster'] = clusters
    data.cluster = (data.cluster + 1).astype('object')
    data.sample(5)

    hp_vs_pri_clu=st.checkbox("Power versus price using clusters")
    if hp_vs_pri_clu:
        fig=plt.figure(figsize=(6,4))
        sns.scatterplot(data=data, y='price', x='power',s=120,palette='viridis',hue='cluster')
        plt.legend(ncol=4)
        plt.title('Scatter plot of price and horsepower with clusterspclusters predicted', fontsize=18)
        plt.xlabel('power',fontsize=16)
        plt.ylabel('price',fontsize=16);
        st.pyplot(fig)

    hp_vs_mil_clu=st.checkbox("Power versus Mileage using clusters")
    if hp_vs_mil_clu:
        fig=plt.figure(figsize=(6,4))
        sns.scatterplot(data=data, x='power', y='mileage',s=120,palette='viridis',hue='cluster')
        plt.legend(ncol=4)
        plt.title('Scatter plot of milage and horsepower with clusters', fontsize=18);
        plt.xlabel('power',fontsize=16)
        plt.ylabel('mileage',fontsize=16);
        st.pyplot(fig)

    eng_vs_fu_clu=st.checkbox("Engine size versus Fuel tank capacity with clusters")
    if eng_vs_fu_clu:
        fig=plt.figure(figsize=(8,6))
        sns.scatterplot(data=data, x='fuel_tank', y='displacement',s=120,palette='viridis',hue='cluster')
        plt.legend(ncol=4)
        plt.title('Scatter plot of Engine size & Fuel tank capacity with clusters', fontsize=18);
        plt.xlabel('Fuel Tank Capacity ',fontsize=16)
        plt.ylabel('Engine size',fontsize=16);
        st.pyplot(fig)

    avg_price=st.checkbox("Average price of each cluster")
    if avg_price:
        fig=plt.figure(figsize=(14,6))
        sns.barplot(data=data, x= 'cluster', ci= 'sd', y= 'price', palette='viridis',order=data.groupby('cluster')['price'].mean().sort_values(ascending=False).index);
        plt.title('Average price of each cluster',fontsize=20)
        plt.xlabel('Cluster',fontsize=16)
        plt.ylabel('Avg car price', fontsize=16)
        plt.xticks(fontsize=14);
        st.pyplot(fig)


    no_of_cars=st.checkbox("Number of cars in each cluster")
    if no_of_cars:
        fig=plt.figure(figsize=(14,4))
        sns.countplot(data=data, x= 'cluster', palette='viridis',order=data.cluster.value_counts().index);
        plt.title('Number of cars in each cluster',fontsize=14)
        plt.xlabel('Cluster',fontsize=16)
        plt.ylabel('Number of cars', fontsize=16)
        plt.xticks(fontsize=14);
        st.pyplot(fig)



    




