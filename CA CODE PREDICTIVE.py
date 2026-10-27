import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Step 1: Load Dataset
os.makedirs("output_charts", exist_ok=True)
sns.set_theme(style="whitegrid")

print("|| STEP 1: LOAD DATASET ||\n")

df = pd.read_csv(r"C:\Users\Hp\Downloads\Supermart_sales.csv")

print("Dataset Shape:", df.shape)
print(df.head(3))


# Step 2: Data Preprocessing and Cleaning
print("\n ||STEP 2: DATA CLEANING ||")

print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

print("\nMissing values:")
print(df.isnull().sum())

df["Order Date"] = pd.to_datetime(
    df["Order Date"], format="mixed", errors="coerce"
)

for col in ["Sales", "Discount", "Profit"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

str_cols = df.select_dtypes(include=["object", "str"]).columns

for col in str_cols:
    df[col] = df[col].astype("str").str.strip()

df.dropna(subset=[
    "Order Date", "Sales", "Discount", "Profit",
    "Category", "Sub Category", "Region"
], inplace=True)

print("\nData types:")
print(df.dtypes)


# Step 3: Feature Engineering
print("\n || STEP 3: FEATURE ENGINEERING ||")

df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_DayOfWeek"] = df["Order Date"].dt.day_name()

df["Profit_Margin_Pct"] = np.where(
    df["Sales"] != 0,
    (df["Profit"] / df["Sales"]) * 100,
    np.nan
)

print(df[
    ["Order Date", "Order_Year", "Order_Month", "Profit_Margin_Pct"]
].head())


# Step 4: Data Analysis and Visualization
print("\n ||STEP 4: FIVE RESEARCH OBJECTIVES ||")


# Objective 1: Sales and Profit by Category
print("\nObjective 1: Sales and Profit by Category")

cat_summary = df.groupby("Category")[["Sales", "Profit"]].sum()
cat_summary = cat_summary.sort_values("Sales", ascending=False)

print(cat_summary)

plt.figure(figsize=(10, 5))
cat_summary.plot(
    kind="bar",
    ax=plt.gca(),
    colormap="viridis"
)

plt.title("Sales and Profit by Category")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("output_charts/obj1_category_sales_profit.png", dpi=300)
plt.show()
plt.close()


# Objective 2: Regional Sales Analysis
print("\nObjective 2: Sales and Profit by Region")

region_summary = df.groupby("Region")[["Sales", "Profit"]].sum()
region_summary = region_summary.sort_values("Sales", ascending=False)

print(region_summary)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=region_summary.reset_index(),
    x="Region",
    y="Sales",
    hue="Region",
    palette="Blues_d",
    legend=False,
    errorbar=None
)

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("output_charts/obj2_region_sales.png", dpi=300)
plt.show()
plt.close()


# Objective 3: Discount and Profit Margin
print("\nObjective 3: Discount and Profit Margin")

plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df,
    x="Discount",
    y="Profit_Margin_Pct",
    alpha=0.5,
    color="darkorange"
)

plt.axhline(0, color="red", linestyle="--")
plt.title("Discount vs Profit Margin")
plt.xlabel("Discount")
plt.ylabel("Profit Margin (%)")
plt.tight_layout()
plt.savefig("output_charts/obj3_discount_vs_margin.png", dpi=300)
plt.show()
plt.close()


# Objective 4: Monthly Sales Trend
print("\nObjective 4: Monthly Sales Trend")

monthly_trend = (
    df.groupby(["Order_Year", "Order_Month"])["Sales"]
    .sum()
    .reset_index()
    .sort_values(["Order_Year", "Order_Month"])
)

monthly_trend["Year_Month"] = (
    monthly_trend["Order_Year"].astype(str)
    + "-"
    + monthly_trend["Order_Month"].astype(str).str.zfill(2)
)

plt.figure(figsize=(12, 5))
plt.plot(
    monthly_trend["Year_Month"],
    monthly_trend["Sales"],
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Year-Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=70)
plt.tight_layout()
plt.savefig("output_charts/obj4_sales_trend.png", dpi=300)
plt.show()
plt.close()


# Objective 5: Top 10 Sub-Categories
print("\nObjective 5: Top 10 Sub-Categories by Sales")

subcat_sales = (
    df.groupby("Sub Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(subcat_sales)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=subcat_sales.values,
    y=subcat_sales.index,
    hue=subcat_sales.index,
    palette="mako",
    legend=False,
    errorbar=None
)

plt.title("Top 10 Sub-Categories by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Sub Category")
plt.tight_layout()
plt.savefig("output_charts/obj5_top_subcategories.png", dpi=300)
plt.show()
plt.close()

print("\nAll five objective charts saved successfully.")


# Step 5: Machine Learning
print("\n || STEP 5: MACHINE LEARNING ||")

X = pd.get_dummies(df[
    ["Category", "Sub Category", "Region", "Discount",
     "Profit", "Order_Year", "Order_Month"]
], drop_first=True)

y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42
    )
}

results = []

for name, model in models.items():

    if name in ["Linear Regression", "Ridge Regression"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    results.append([
        name,
        round(mean_absolute_error(y_test, predictions), 2),
        round(np.sqrt(mean_squared_error(y_test, predictions)), 2),
        round(r2_score(y_test, predictions), 4)
    ])

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2 Score"]
)

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

best_model = results_df.loc[results_df["R2 Score"].idxmax()]

print("\nBest Model:", best_model["Model"])
print("R2 Score:", best_model["R2 Score"])


# Step 6: Model Comparison Chart
plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score",
    hue="Model",
    legend=False,
    palette="crest",
    errorbar=None
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("output_charts/model_performance_comparison.png", dpi=300)
plt.show()
plt.close()

print("Project completed successfully.")
