import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Covid-19.csv")

# print(data.head(5))
print("Columns in the Dataset:")
print(data.columns)

print("\n Dataset shape:", data.shape)

print("Statistical Summary:")
print(data.describe)

print("\n Missing Values:")
print(data.isnull().sum())

cols_to_fix = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Active Cases', 'Population']

for col in cols_to_fix:
    if col in data.columns:
        # Remove commas and convert to numeric
        data[col] = data[col].astype(str).str.replace(',', '').str.replace(' ', '')
        data[col] = pd.to_numeric(data[col], errors='coerce')

print("\n Dtypes After cleaning:")
print(data.dtypes)

top10_cases = data.nlargest(10, 'Total Cases')

plt.figure(figsize=(10,6))
sns.barplot(x='Total Cases', y='Country', data=top10_cases, palette= 'Reds_r')
plt.title("Top 10 Countries by total Covid-19 Cases", fontsize=14)
plt.xlabel("Total Cases")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

sns.relplot(x="Population", y="Total Cases", hue="Total Recovered", size="Active Cases",
            data=top10_cases, sizes=(50, 400), palette="viridis")
plt.title("COVID-19: Population vs Total Cases (Top 10 Countries)")
plt.tight_layout()
plt.show()

# --- C. Pairplot for overall relationships ---
sns.pairplot(data[['Total Cases', 'Total Deaths', 'Total Recovered', 'Active Cases']], corner=True)
plt.suptitle("Feature Relationships in COVID-19 Data", y=1.02)
plt.show()


# --- D. Trend between Active and Total Cases ---
sns.relplot(x='Total Cases', y='Active Cases', kind='line', data=data.sort_values('Total Cases'))
plt.title("Trend: Active Cases vs Total Cases")
plt.tight_layout()
plt.show()

# Machine Learning prep starts from here 

data.dropna(subset=cols_to_fix, inplace=True)

X = data[['Total Cases', 'Total Recovered', 'Active Cases', 'Population']]
y = data['Total Deaths']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🎯 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.3f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color='teal', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Actual vs Predicted COVID-19 Deaths")
plt.tight_layout()
plt.show()

importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)


print("\n Feature Importance DataFrame:")
print(feature_importance)

# Ensure correct column names
feature_importance.columns = ['Feature', 'Importance']

# Visualization
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title("Feature Importance - Random Forest Model")
plt.tight_layout()
plt.show()

# STREAMLIT Dashboard making

st.markdown("---")
st.header(" Interactive COVID-19 Death Prediction Tool")
st.markdown("Use this interactive tool to estimate **Total Deaths** based on your custom COVID-19 scenario.")

with st.expander("🔍 Click to open the interactive predictor"):
    st.write("Adjust the sliders and inputs below to simulate different pandemic conditions and see how they affect predicted deaths.")

    # Split UI into two columns for cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        total_cases = st.slider("Total Cases", 0, 10_000_000, 100_000, step=5_000)
        total_recovered = st.slider("Total Recovered", 0, 10_000_000, 80_000, step=5_000)
        active_cases = st.slider("Active Cases", 0, 1_000_000, 15_000, step=1_000)

    with col2:
        population = st.number_input("Population", min_value=1_000, max_value=2_000_000_000, value=1_000_000, step=50_000)
        vaccination_rate = st.slider("Vaccination Rate (%)", 0, 100, 60)
        positivity_rate = st.slider("Positivity Rate (%)", 0, 100, 10)

    # Prepare the user input DataFrame
    user_input = pd.DataFrame({
        'Total Cases': [total_cases],
        'Total Recovered': [total_recovered],
        'Active Cases': [active_cases],
        'Population': [population],
        'Vaccination Rate': [vaccination_rate],
        'Positivity Rate': [positivity_rate]
    })

    st.markdown("Input Summary")
    st.dataframe(user_input.style.highlight_max(color='lightgreen', axis=1))

    # Predict button
    if st.button(" Predict Deaths Now"):
        try:
            # Match training feature columns
            missing_cols = [col for col in X.columns if col not in user_input.columns]
            for col in missing_cols:
                user_input[col] = 0  # placeholder if not used

            # Scale and predict
            user_input_scaled = scaler.transform(user_input[X.columns])
            prediction = model.predict(user_input_scaled)[0]

            # Display results
            st.success(f" **Predicted Total Deaths: {prediction:,.0f}**")

            # Show contextual feedback
            st.markdown(f"""
            -  Population Size: **{population:,}**
            -  Vaccination Rate: **{vaccination_rate}%**
            -  Positivity Rate: **{positivity_rate}%**
            """)

            # Visual comparison graph
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            # Pie chart showing proportion of deaths
            death_rate = (prediction / population) * 100
            ax[0].pie(
                [death_rate, 100 - death_rate],
                labels=["Predicted Deaths (%)", "Surviving Population (%)"],
                autopct="%.2f%%",
                startangle=90,
                colors=["#ff4b4b", "#4caf50"]
            )
            ax[0].set_title("Predicted Mortality Share")

            # Horizontal bar showing user inputs
            sns.barplot(
                y=["Total Cases", "Recovered", "Active Cases"],
                x=[total_cases, total_recovered, active_cases],
                ax=ax[1],
                palette="viridis"
            )
            ax[1].set_title("Input Summary")
            ax[1].set_xlabel("Count")

            st.pyplot(fig)

            # Save results (optional)
            if "prediction_history" not in st.session_state:
                st.session_state["prediction_history"] = pd.DataFrame(columns=user_input.columns.tolist() + ["Predicted Deaths"])
            new_row = user_input.copy()
            new_row["Predicted Deaths"] = prediction
            st.session_state["prediction_history"] = pd.concat([st.session_state["prediction_history"], new_row], ignore_index=True)

            st.markdown("Prediction History")
            st.dataframe(st.session_state["prediction_history"].tail(5))

            # Line chart to show predictions over time
            st.line_chart(st.session_state["prediction_history"]["Predicted Deaths"], use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while predicting: {e}")