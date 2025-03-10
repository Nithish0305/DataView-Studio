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
        # Allow the user to select columns for Covariance Matrix calculation
        selected_columns = st.multiselect(
            "Select columns for Covariance Matrix computation:",
            options=data.columns.tolist(),
            default=data.columns.tolist()
        )

        if not selected_columns:
            st.warning("Please select at least one column to compute the Covariance Matrix.")
            return

        # Filter the data based on selected columns
        selected_data = data[selected_columns]

        # Check for non-numeric columns
        if not selected_data.apply(lambda col: np.issubdtype(col.dtype, np.number)).all():
            non_numeric_columns = selected_data.select_dtypes(exclude=np.number).columns.tolist()
            st.error(f"The following columns contain non-numeric data and cannot be used for the Covariance Matrix: {', '.join(non_numeric_columns)}")
            return

        # Display the Covariance Matrix
        st.write("Covariance Matrix:")
        st.write(selected_data.cov())

    elif stats_option == "Correlation Matrix":
        # Allow the user to select columns for Correlation Matrix calculation
        selected_columns = st.multiselect(
            "Select columns for Correlation Matrix computation:",
            options=data.columns.tolist(),
            default=data.columns.tolist()
        )

        if not selected_columns:
            st.warning("Please select at least one column to compute the Correlation Matrix.")
            return

        # Filter the data based on selected columns
        selected_data = data[selected_columns]

        # Check for non-numeric columns
        if not selected_data.apply(lambda col: np.issubdtype(col.dtype, np.number)).all():
            non_numeric_columns = selected_data.select_dtypes(exclude=np.number).columns.tolist()
            st.error(f"The following columns contain non-numeric data and cannot be used for the Correlation Matrix: {', '.join(non_numeric_columns)}")
            return

        # Display the Correlation Matrix
        st.write("Correlation Coefficient Matrix:")
        st.write(selected_data.corr())


def calculate_pca(data):
    st.subheader("PCA")

    # Allow the user to select columns for PCA
    selected_columns = st.multiselect(
        "Select columns for PCA computation:",
        options=data.columns.tolist(),
        default=data.columns.tolist()
    )

    if not selected_columns:
        st.warning("Please select at least one column for PCA.")
        return

    # Filter the data based on selected columns
    selected_data = data[selected_columns]

    # Check for non-numeric columns
    if not selected_data.apply(lambda col: np.issubdtype(col.dtype, np.number)).all():
        non_numeric_columns = selected_data.select_dtypes(exclude=np.number).columns.tolist()
        st.error(f"The following columns contain non-numeric data and cannot be used for PCA: {', '.join(non_numeric_columns)}")
        return

    # Display the selected data for transparency
    st.write("Selected data for PCA:")
    st.dataframe(selected_data)

    pca_option = st.session_state.get('pca_option', None)

    if pca_option == "Principal Component Analysis":
        scaler = StandardScaler()
        toggle = st.checkbox("Standardize")

        # Standardize or directly use the selected data
        if toggle:
            pca_data = scaler.fit_transform(selected_data)
        else:
            pca_data = selected_data.values - np.mean(selected_data.values, axis=0)  # Center data if not standardizing

        # Display covariance matrix
        st.write("Covariance Matrix:")
        st.write(pd.DataFrame(np.cov(pca_data, rowvar=False), columns=selected_columns, index=selected_columns))

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
        st.write(pd.DataFrame(pca.components_, columns=selected_columns, index=[f"PC{i+1}" for i in range(len(eigenvalues))]))

def calculate_linearregression_predict(data=None):
    linearregression_option = st.session_state.get('linearregression_option', None)
    
    if linearregression_option == "OLS":
        st.subheader("Multiple Linear Regression Prediction")

        if data is not None:
            if data.shape[1] < 2:
                st.error("Data must contain at least one independent variable and one dependent variable.")
                return

            # Allow user to select independent and dependent variables
            st.write("### Select Independent and Dependent Variables")
            all_columns = list(data.select_dtypes(include="number"))
            selected_features = st.multiselect("Select feature columns (independent variables)", all_columns, default=all_columns[:-1])
            selected_target = st.selectbox("Select target column (dependent variable)", all_columns, index=len(all_columns) - 1)

            if not selected_features or not selected_target:
                st.error("Please select at least one feature and one target column.")
                return

            # Extract selected features and target variable
            X = data[selected_features].values
            y = data[selected_target].values
            n = len(y)

            # Add a column of ones for the intercept term (b0)
            X_matrix = np.c_[np.ones(n), X]

            # Compute normal equation components
            XT_X = np.dot(X_matrix.T, X_matrix)  # (X^T * X)
            XT_y = np.dot(X_matrix.T, y)  # (X^T * y)

            try:
                # Solve for coefficients using normal equations
                coef = np.linalg.solve(XT_X, XT_y)

                # Display coefficients
                st.write("### Calculated Coefficients:")
                coef_dict = {f"b{i} (for {col} if applicable)": coef[i] for i, col in enumerate(["Intercept"] + selected_features)}
                st.write(coef_dict)

                # User input for prediction
                st.write("### Predict New Value")
                x_input = [st.number_input(f"Enter value for {col}", value=0.0) for col in selected_features]
                prediction = np.dot(np.insert(x_input, 0, 1), coef)
                st.write(f"**Predicted y value:** {prediction}")

            except np.linalg.LinAlgError:
                st.error("Error: Unable to compute coefficients. The matrix may be singular.")

        else:
            # Manual input for summations when dataset is not provided
            num_features = st.number_input("Enter number of independent variables", min_value=1, value=1)

            # Initialize lists to store summation inputs
            sum_x = []
            sum_xx = []
            sum_xy = []

            for i in range(num_features):
                sum_x.append(st.number_input(f"Σx{i+1}", value=0.0))
                sum_xx.append(st.number_input(f"Σx{i+1}²", value=0.0))
                sum_xy.append(st.number_input(f"Σx{i+1} * y", value=0.0))

            sum_y = st.number_input("Σy", value=0.0)
            n = st.number_input("Enter number of data points (n)", min_value=1, value=1)

            # Construct normal equation matrix
            A = np.zeros((num_features + 1, num_features + 1))
            B = np.zeros(num_features + 1)

            A[0, 0] = n
            B[0] = sum_y

            for i in range(num_features):
                A[0, i + 1] = A[i + 1, 0] = sum_x[i]
                A[i + 1, i + 1] = sum_xx[i]
                B[i + 1] = sum_xy[i]

                for j in range(i + 1, num_features):
                    A[i + 1, j + 1] = A[j + 1, i + 1] = st.number_input(f"Σx{i+1} * x{j+1}", value=0.0)

            try:
                # Solve for coefficients
                coefficients = np.linalg.solve(A, B)

                st.write("Calculated Coefficients:")
                st.write({f"b{i}": coefficients[i] for i in range(len(coefficients))})

                # User input for prediction
                x_input = [st.number_input(f"Enter value for x{i+1}", value=0.0) for i in range(num_features)]
                prediction = coefficients[0] + np.dot(coefficients[1:], x_input)
                st.write(f"Predicted y value: {prediction}")

            except np.linalg.LinAlgError:
                st.error("Error: Unable to compute coefficients. The matrix may be singular.")

    elif linearregression_option == "WLS":
        st.subheader("Weighted Least Squares Regression Prediction")

        if data is not None:
            if data.shape[1] < 2:
                st.error("Data must contain at least one independent variable and one dependent variable.")
                return

            st.write("### Select Independent and Dependent Variables")
            all_columns = list(data.select_dtypes(include="number"))
            selected_features = st.multiselect("Select feature columns (independent variables)", all_columns, default=all_columns[:-1])
            selected_target = st.selectbox("Select target column (dependent variable)", all_columns, index=len(all_columns) - 1)

            if not selected_features or not selected_target:
                st.error("Please select at least one feature and one target column.")
                return

            X = data[selected_features].values
            y = data[selected_target].values
            n = len(y)

            X_matrix = np.c_[np.ones(n), X]
            XT_X = np.dot(X_matrix.T, X_matrix)
            XT_y = np.dot(X_matrix.T, y)

            try:
                ols_coef = np.linalg.solve(XT_X, XT_y)
                residuals = y - np.dot(X_matrix, ols_coef)
                weights = 1 / (residuals**2 + 1e-8)
                
                st.write("### Computed Weights:")
                st.write(weights)
                
                W = np.diag(weights)
                XTWX = np.dot(X_matrix.T, np.dot(W, X_matrix))
                XTWY = np.dot(X_matrix.T, np.dot(W, y))
                wls_coef = np.linalg.solve(XTWX, XTWY)

                st.write("### Calculated Coefficients (WLS):")
                coef_dict = {f"b{i} (for {col} if applicable)": wls_coef[i] for i, col in enumerate(["Intercept"] + selected_features)}
                st.write(coef_dict)

                st.write("### Predict New Value")
                x_input = [st.number_input(f"Enter value for {col}", value=0.0) for col in selected_features]
                prediction = np.dot(np.insert(x_input, 0, 1), wls_coef)
                st.write(f"**Predicted y value:** {prediction}")
            except np.linalg.LinAlgError:
                st.error("Error: Unable to compute coefficients. The matrix may be singular.")


# Main function to handle different app sections
def main():
    st.markdown('<h1 style=color:rgb(0,200,200)>DataView Studio</h1>', unsafe_allow_html=True)
    st.header('Upload your file here to get insights')

    # File uploader
    data = st.file_uploader("Upload your data here", type=["xlsx", "xls"])

    # Display hamburger icon and Main Menu text in sidebar
    with st.sidebar:
        st.markdown("<div style='font-size: 30px; cursor: pointer;'>&#9776; Main Menu</div>", unsafe_allow_html=True)

        # Initialize session state for menu selections if not already present
        if 'current_menu' not in st.session_state:
            st.session_state.current_menu = "Linear Regression"
        
        # Sidebar menu with nested structure
        selected = option_menu(
            menu_title=None,
            options=["Linear Regression", "Visualizations", "Descriptive Statistics", "PCA"],
            icons=[None, "graph-up", "calculator", "diagram-3"],
            menu_icon="cast",
            default_index=0,
        )
        
        # Update current_menu based on selection
        st.session_state.current_menu = selected
        
        # Submenu based on main selection
        if selected == "Visualizations":
            viz_option = option_menu(
                menu_title=None,
                options=["Line Plot", "Scatter Plot", "Histogram", "Box Plot", 
                         "Bar Plot", "Violin Plot", "Count Plot"],
                icons=["graph-up", "scatter-chart", "bar-chart-steps", "box", 
                       "bar-chart", "vinyl", "stack"],
            )
            st.session_state.visualization_option = viz_option
            st.session_state.stats_option = None
            st.session_state.pca_option = None
            st.session_state.linearregression_option = None
            
        elif selected == "Descriptive Statistics":
            stats_option = option_menu(
                menu_title=None,
                options=["Mean", "Median", "Mode", "Variance", "Standard Deviation", "Minimum", "Maximum", 
                         "Covariance Matrix", "Correlation Matrix"],
            )
            st.session_state.stats_option = stats_option
            st.session_state.visualization_option = None
            st.session_state.pca_option = None
            st.session_state.linearregression_option = None
            
        elif selected == "PCA":
            pca_option = option_menu(
                menu_title=None,
                options=["Principal Component Analysis"]
            )
            st.session_state.pca_option = pca_option
            st.session_state.visualization_option = None
            st.session_state.stats_option = None
            st.session_state.linearregression_option = None
            
        elif selected == "Linear Regression":
            linearregression_option = option_menu(
                menu_title=None,
                options=["OLS","WLS"]
            )
            st.session_state.linearregression_option = linearregression_option
            st.session_state.visualization_option = None 
            st.session_state.pca_option = None
            st.session_state.stats_option = None

    # Main content area
    if data is not None:
        st.success("File Uploaded Successfully")
        data1 = pd.read_excel(data)
        
        # Call the appropriate function based on the current menu selection
        if st.session_state.current_menu == "Linear Regression":
            calculate_linearregression_predict(data1)
        elif st.session_state.current_menu == "Visualizations":
            visualize_data(data1)
        elif st.session_state.current_menu == "Descriptive Statistics":
            calculate_statistics(data1)
        elif st.session_state.current_menu == "PCA":
            calculate_pca(data1)
    else:
        if st.session_state.current_menu == "Linear Regression":
            calculate_linearregression_predict()
        else:
            st.warning("Please upload a file to proceed with the selected analysis.")

if __name__ == '__main__':
    main()
