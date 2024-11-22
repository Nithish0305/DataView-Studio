import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

def create_line_plot(data, x_column, y_column):
    if x_column in data.columns and y_column in data.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='lines+markers'))
        fig.update_layout(
            title=f'{y_column} vs {x_column}',
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif x_column == "Select a column" or y_column == "Select a column":
        st.warning("Please select the required fields")

# Function to create Scatter Plot
def create_scatter_plot(data, x_column, y_column):
    if x_column in data.columns and y_column in data.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='markers'))
        fig.update_layout(
            title=f'{y_column} vs {x_column} Scatter Plot',
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif x_column == "Select a column" or y_column == "Select a column":
        st.warning("Please select the required fields")

# Function to create Histogram
def create_histogram(data, column):
    if column in data.columns:
        fig = px.histogram(data, x=column, nbins=30, title=f'Histogram of {column}')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif column == "Select a column":
        st.warning("Please select a column")

# Function to create Box Plot
def create_box_plot(data, column):
    if column in data.columns:
        fig = px.box(data, y=column, title=f'Box Plot of {column}')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif column == "Select a column":
        st.warning("Please select a column")

# Function to create Bar Plot
def create_bar_plot(data, x_column, y_column):
    if x_column in data.columns and y_column in data.columns:
        fig = px.bar(data, x=x_column, y=y_column, title=f'Bar Plot of {y_column} vs {x_column}')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif x_column == "Select a column" or y_column == "Select a column":
        st.warning("Please select the required fields")

# Function to create Violin Plot
def create_violin_plot(data, column):
    if column in data.columns:
        fig = px.violin(data, y=column, box=True, points="all", title=f'Violin Plot of {column}')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif column == "Select a column":
        st.warning("Please select a column")

# Function to create Count Plot
def create_count_plot(data, column):
    if column in data.columns:
        fig = px.bar(data, x=column, title=f'Count Plot of {column}')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif column == "Select a column":
        st.warning("Please select a column")

# Main visualization function
def visualize_data(data):
    st.subheader("Visualize the Data")
    
    columns = ["Select a column"]
    columns.extend(data.columns)
    
    # Get the selected visualization type from session state
    visualization_option = st.session_state.get('visualization_option', None)
    
    if visualization_option == "Line Plot":
        x_column = st.selectbox("Select X-axis column", columns, key='xcol')
        y_column = st.selectbox("Select Y-axis column", columns, key='ycol')
        create_line_plot(data, x_column, y_column)
        
    elif visualization_option == "Scatter Plot":
        x_column = st.selectbox("Select X-axis column", columns, key='xcol_scatter')
        y_column = st.selectbox("Select Y-axis column", columns, key='ycol_scatter')
        create_scatter_plot(data, x_column, y_column)
        
    elif visualization_option == "Histogram":
        column = st.selectbox("Select a column", columns, key='hist_col')
        create_histogram(data, column)
        
    elif visualization_option == "Box Plot":
        column = st.selectbox("Select a column", columns, key='box_col')
        create_box_plot(data, column)
        
    elif visualization_option == "Bar Plot":
        x_column = st.selectbox("Select X-axis column", columns, key='xcol_bar')
        y_column = st.selectbox("Select Y-axis column", columns, key='ycol_bar')
        create_bar_plot(data, x_column, y_column)
        
    elif visualization_option == "Violin Plot":
        column = st.selectbox("Select a column", columns, key='violin_col')
        create_violin_plot(data, column)
        
    elif visualization_option == "Count Plot":
        column = st.selectbox("Select a column", columns, key='count_col')
        create_count_plot(data, column)

def calculate_statistics(data):
    st.subheader("Descriptive Statistics")
    
    columns = ["Select a column"]
    columns.extend(data.columns)
    
    # Get the selected statistics option from session state
    stats_option = st.session_state.get('stats_option', None)
    
    if stats_option == "Mean":
        mean_column = st.selectbox("Select column to find Mean", columns, key='meancol')
        if mean_column in data.columns:
            mean = np.mean(data[mean_column])
            st.write(f"The average of the column '{mean_column}' is {mean}")
        elif mean_column == "Select a column":
            st.warning("Please select the required fields")
    elif stats_option == "Median":
        median_column = st.selectbox("Select column to find Median", columns, key='mediancol')
        if median_column in data.columns:
            median = np.median(data[median_column])
            st.write(f"The median of the column '{median_column}' is {median}")
        elif median_column == "Select a column":
            st.warning("Please select the required fields")

    elif stats_option == "Mode":
        mode_column = st.selectbox("Select column to find Mode", columns, key='modecol')
        if mode_column in data.columns:
            mode = stats.mode(data[mode_column])
            st.write(f"Mode: {mode.mode}, Count: {mode.count}")
        elif mode_column == "Select a column":
            st.warning("Please select the required fields")
    elif stats_option == "Minimum":
        minimum_column = st.selectbox("Select column to find Minimum", columns, key='mincol')
        if minimum_column in data.columns:
            minimum = data[minimum_column].min()
            st.write(f"The Minimum of the column '{minimum_column}' is {minimum}")
        elif minimum_column == "Select a column":
            st.warning("Please select the required fields") 

    elif stats_option == "Maximum":
        maximum_column = st.selectbox("Select column to find Maximum", columns, key='maxcol')
        if maximum_column in data.columns:
            maximum = data[maximum_column].max()
            st.write(f"The Maxmimum of the column '{maximum_column}' is {maximum}")
        elif maximum_column == "Select a column":
            st.warning("Please select the required fields")                  

    elif stats_option == "Standard Deviation":
        sd_column = st.selectbox("Select column to find Standard Deviation", columns, key='sdcol')
        if sd_column in data.columns:
            sd = np.std(data[sd_column])
            st.write(f"The Standard Deviation of the column '{sd_column}' is {sd}")
        elif sd_column == "Select a column":
            st.warning("Please select the required fields")                         
            
    elif stats_option == "Variance":
        var_column = st.selectbox("Select column to find variance", columns, key='varcol')
        if var_column in data.columns:
            variance = np.var(data[var_column])
            st.write(f"The variance of the column '{var_column}' is {variance}")
        elif var_column == "Select a column":
            st.warning("Please select the required fields")
            
    elif stats_option == "Covariance Matrix":
        st.write("Covariance Matrix:")
        st.write(data.cov())
        
    elif stats_option == "Correlation Matrix":
        st.write("Correlation Coefficient Matrix:")
        st.write(data.corr())

def calculate_pca(data):
    st.subheader("PCA")
    pca_option = st.session_state.get('pca_option', None)

    if pca_option == "Principal Component Analysis":
        scaler = StandardScaler()
        toggle = st.checkbox("Standardize")

        # Standardize or directly use the data
        if toggle:
            pca_data = scaler.fit_transform(data)
        else:
            pca_data = data.values - np.mean(data.values, axis=0)  # Center data if not standardizing

        # Display covariance matrix
        st.write("Covariance Matrix:")
        st.write(pd.DataFrame(np.cov(pca_data, rowvar=False), columns=data.columns, index=data.columns))

        # PCA calculation
        pca = PCA()
        pca.fit(pca_data)
        pca_matrix = pca.transform(pca_data)

        # Eigenvalues and proportions
        eigenvalues = pca.explained_variance_
        proportions = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(proportions)

        # Create a scree plot with Plotly
        fig = go.Figure()

        # Bar chart for eigenvalues
        fig.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(eigenvalues))],
            y=eigenvalues,
            name="Eigenvalue",
            text=[f"{ev:.4f}" for ev in eigenvalues],
            textposition="auto"
        ))

        # Line plot for cumulative variance
        fig.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
            y=cumulative_variance,
            name="Cumulative Variance (%)",
            mode="lines+markers",
            marker=dict(color="orange")
        ))

        # Customize layout
        fig.update_layout(
            title="Scree Plot with Cumulative Variance",
            xaxis_title="Component",
            yaxis=dict(title="Eigenvalue"),
            yaxis2=dict(title="Cumulative Variance (%)", overlaying='y', side='right'),
            legend=dict(x=0.1, y=1.1),
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Display Eigenvalues and Proportions as a DataFrame
        eigvalues_proportions = pd.DataFrame({
            "Eigenvalues": eigenvalues,
            "Proportion (%)": proportions
        })
        st.write("Eigenvalues and Proportions:")
        st.dataframe(eigvalues_proportions)

        # Display Principal Components
        st.write("Principal Components/Scores (Tilted Data):")
        st.write(pd.DataFrame(pca_matrix, columns=[f"PC{i+1}" for i in range(pca_matrix.shape[1])]))

        # Display PCA Components (Eigenvectors)
        st.write("PCA Components (Eigenvectors):")
        st.write(pd.DataFrame(pca.components_, columns=data.columns, index=[f"PC{i+1}" for i in range(len(eigenvalues))]))



def main():
    
    st.markdown('<h1 style=color:rgb(0,200,200)>DataView Studio</h1>', unsafe_allow_html=True)
    st.header('Upload your file here to get insights')
    
    # File uploader
    data = st.file_uploader("Upload your data here", type=["xlsx", "xls"])

    # Display hamburger icon and Main Menu text in sidebar
    with st.sidebar:
        # Use HTML to create a hamburger menu icon with "Main Menu" label
        st.markdown("<div style='font-size: 30px; cursor: pointer;'>&#9776; Main Menu</div>", unsafe_allow_html=True)

        # Initialize session state for menu selections if not already present
        if 'current_menu' not in st.session_state:
            st.session_state.current_menu = "Visualizations"
        
        # Sidebar menu with nested structure
        selected = option_menu(
            menu_title=None,
            options=["Visualizations", "Descriptive Statistics", "PCA"],
            icons=["graph-up", "calculator", "diagram-3"],
            menu_icon="cast",  # Icon for the main menu
            default_index=0,
        )
        
        # Submenu based on main selection
        if selected == "Visualizations":
            st.session_state.current_menu = "Visualizations"
            viz_option = option_menu(
                menu_title=None,
                options=["Line Plot", "Scatter Plot", "Histogram", "Box Plot", 
                         "Bar Plot", "Violin Plot", "Count Plot"],
                icons=["graph-up", "scatter-chart", "bar-chart-steps", "box", 
                       "bar-chart", "vinyl", "stack"],
            )
            st.session_state.visualization_option = viz_option
            st.session_state.stats_option = None
            
        elif selected == "Descriptive Statistics":
            st.session_state.current_menu = "Descriptive Statistics"
            stats_option = option_menu(
                menu_title=None,
                options=["Mean", "Median", "Mode","Variance","Standard Deviation","Minimum","Maximum", 
                         "Covariance Matrix", "Correlation Matrix"],
         
            )
            st.session_state.stats_option = stats_option
            st.session_state.visualization_option = None
        elif selected == "PCA":
            st.session_state.current_menu = "PCA"
            pca_option = option_menu(
                menu_title=None,
                options=["Principal Component Analysis"]
            )
            st.session_state.pca_option = pca_option
            st.session_state.visualization_option = None    

    # Main content area
    if data is not None:
        st.success("File Uploaded Successfully")
        data1 = pd.read_excel(data)
        st.write("Preview of the data:")
        st.write(data1.head())
        
        if st.session_state.current_menu == "Visualizations":
            visualize_data(data1)
        elif st.session_state.current_menu == "Descriptive Statistics":
            calculate_statistics(data1)
        elif st.session_state.current_menu == "PCA":
            calculate_pca(data1)


if __name__ == "__main__":
    main()