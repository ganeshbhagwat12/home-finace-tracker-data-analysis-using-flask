# app.py
from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import datetime
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load income and expense data (replace with your actual data loading logic)
income_df = pd.read_csv('income.csv')
expense_df = pd.read_csv('exp.csv')
budget_df = pd.read_csv('budget.csv')

income_df = income_df.dropna()  # Remove rows with missing values
expense_df = expense_df.dropna()

income_df = income_df.drop_duplicates()
expense_df = expense_df.drop_duplicates()

budget_df= budget_df.dropna()
budget_df = budget_df.drop_duplicates()
budget_df['Date'] = pd.to_datetime(budget_df['Date'], format='%d-%m-%Y')

scaler = MinMaxScaler()

# Normalize numerical columns in income_df
income_df[income_df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(income_df.select_dtypes(include=['float64', 'int64']))

# Normalize numerical columns in expense_df
expense_df[expense_df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(expense_df.select_dtypes(include=['float64', 'int64']))

# Normalize numerical columns in budget_df
budget_df[budget_df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(budget_df.select_dtypes(include=['float64', 'int64']))

X_budget = budget_df[['amount', 'Savings']]
y_budget = budget_df['Amount (INR)']

# Split the budget data into training and test sets (80% training, 20% test)
X_budget_train, X_budget_test, y_budget_train, y_budget_test = train_test_split(X_budget, y_budget, test_size=0.2, random_state=42)

# Train the linear regression model on budget data
regression_model = LinearRegression()
regression_model.fit(X_budget_train, y_budget_train)

# Make predictions on the test set for budget data
y_budget_pred = regression_model.predict(X_budget_test)

# Combine actual and predicted values for budget data into a DataFrame
budget_comparison_df = pd.DataFrame({'Actual': y_budget_test, 'Predicted': y_budget_pred})

# Now, assuming income_df is defined earlier, predict income using the trained model
income_df['Predicted_Income'] = regression_model.predict(income_df[['amount', 'Savings']])

# Calculate the total actual income and predicted income
total_actual_income = income_df['amount'].sum()
total_predicted_income = income_df['Predicted_Income'].sum()

# Assuming expance_df and income_df are defined earlier
# Calculate the total actual expenses and predicted expenses
total_actual_expenses = expense_df['Amount (INR)'].sum()
total_predicted_expenses = expense_df['Amount (INR)'].sum()

def plot_to_base64(fig):
    # Convert the plot to base64 to embed in HTML
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return plot_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/income')
def income_analysis():
    # Ensure 'amount' column contains numeric values
    income_df['amount'] = pd.to_numeric(income_df['amount'], errors='coerce')
    income_df['amount'].fillna(0, inplace=True)

    # Total income
    total_income = income_df['amount'].sum()

    # Income by categories (assuming the correct column name is 'categories')
    income_by_categories = income_df.groupby('Categories')['amount'].sum().reset_index()
    income_df['Categories'] = income_df['Categories'].astype(str)

    # Profit margin analysis
    Profit_Margin_fig, ax = plt.subplots()
    ax.bar(income_df['Categories'], income_df['Savings'])
    ax.set_title('Savings Analysis')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Savings (%)')
    plt.xticks(rotation=45, ha='right')  # Rotate category labels for better visibility
    Profit_Margin_chart = plot_to_base64(Profit_Margin_fig)

    # Trending chart (you need to replace 'date' with your actual time column)
    Trending_fig, ax = plt.subplots()
    ax.plot(income_df['amount'], marker='o', linestyle='-')
    ax.set_title('Income Trend Over Time')
    ax.set_ylabel('Income')
    plt.xticks(rotation=45, ha='right')
    Trending_chart = plot_to_base64(Trending_fig)

    # Most income category
    most_income_category = income_by_categories.loc[income_by_categories['amount'].idxmax()]
    
    income_by_categories = income_by_categories.sort_values(by='amount', ascending=False)

    # Divide income categories into top 50% and bottom 50%
    top_50_percent = income_by_categories.head(len(income_by_categories) // 2)
    bottom_50_percent = income_by_categories.tail(len(income_by_categories) // 2)

    # Pie chart for top 50% income categories
    Top_Pie_chart_fig, ax = plt.subplots()
    ax.pie(top_50_percent['amount'], labels=top_50_percent['Categories'], autopct='%1.0f%%', startangle=360)
    ax.axis('equal')
    ax.set_title('Top 50% Income Distribution by Categories')
    Top_Pie_chart = plot_to_base64(Top_Pie_chart_fig)

    # Pie chart for bottom 50% income categories
    Bottom_Pie_chart_fig, ax = plt.subplots()
    ax.pie(bottom_50_percent['amount'], labels=bottom_50_percent['Categories'], autopct='%1.0f%%', startangle=360)
    ax.axis('equal')
    ax.set_title('Bottom 50% Income Distribution by Categories')
    Bottom_Pie_chart = plot_to_base64(Bottom_Pie_chart_fig)

    mean_income = income_df['Savings'].mean()
    median_income = income_df['Savings'].median()
    std_dev_income = income_df['Savings'].std()

    plt.figure(figsize=(8, 6))
    sns.heatmap(income_df[['Savings']], cmap='viridis', annot=True, fmt=".2f", cbar_kws={'label': 'Profit Margin'})
    plt.title('Heatmap of Savings')
    plt.xlabel('Data Points')
    plt.ylabel('Savings')
    plt.tight_layout()
    img_data_heatmap = plot_to_base64(plt.gcf())

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Savings', y='amount', data=income_df)
    plt.title('Income Summary of analysis')
    plt.xlabel('Savings')
    plt.ylabel('amount')
    summary_chart = plot_to_base64(plt.gcf())
    
    return render_template('income.html',
                           total_income=total_income,
                           income_by_categories=income_by_categories.to_html(),
                           Profit_Margin_chart=Profit_Margin_chart,
                           Trending_chart=Trending_chart,
                           most_income_category=most_income_category,Top_Pie_chart=Top_Pie_chart,
                           Bottom_Pie_chart=Bottom_Pie_chart,mean_income=mean_income, median_income=median_income, 
                           std_dev_income=std_dev_income,img_data_heatmap=img_data_heatmap,summary_chart=summary_chart)
@app.route('/expense')
def expense_analysis():
    # Total expenses
    total_expense = expense_df['Amount (INR)'].sum()

    # Expenses by category
    expense_by_category = expense_df.groupby('Category')['Amount (INR)'].sum().reset_index()


    # Expense trends over time
    expense_trends_fig, ax = plt.subplots()
    ax.plot(expense_df['Date'], expense_df['Amount (INR)'])
    ax.set_title('Expense Trends Over Time')
    expense_trends_chart = plot_to_base64(expense_trends_fig)

    pie_chart_fig, ax = plt.subplots(figsize=(8, 8))

# Plot the pie chart
    ax.pie(expense_by_category['Amount (INR)'], labels=expense_by_category['Category'], autopct='%1.0f%%', startangle=460)

    # Set title for the pie chart
    ax.set_title('Expense Distribution by Category')

    # Assuming plot_to_base64 function is defined somewhere else in your code
    # pie_chart = plot_to_base64(pie_chart_fig)
    pie_chart = plot_to_base64(pie_chart_fig)
    
    # Convert the 'date' column to datetime type
    expense_df['Date'] = pd.to_datetime(expense_df['Date'])

    # Extract the month from the 'date' column
    expense_df['month'] = expense_df['Date'].dt.month_name()

    # Assuming your CSV has columns like 'date' and 'expense'
    # Modify the column names accordingly
    top_expenses_data = expense_df.groupby('month')['Amount (INR)'].sum().sort_values(ascending=False).head(10)
    top_expenses_fig, ax = plt.subplots()
    top_expenses_data.plot(kind='bar', ax=ax)
    ax.set_title('Monthly Expenses')
    ax.set_xlabel('Month')
    ax.set_ylabel('Expense')

    # Convert the plot to base64
    top_expenses_monthly = plot_to_base64(top_expenses_fig)

    # Top expenses
    top_expenses_fig, ax = plt.subplots()
    ax.bar(expense_df.nlargest(5, 'Amount (INR)')['Item'], expense_df.nlargest(5, 'Amount (INR)')['Amount (INR)'])
    ax.set_title('Top Expenses Overal')
    top_expenses_chart = plot_to_base64(top_expenses_fig)
     
    monthly_expenses_pivot = expense_df.pivot_table(index='month', aggfunc='sum', values='Amount (INR)')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(monthly_expenses_pivot, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=.5, cbar_kws={'label': 'Expense (INR)'})
    plt.title('Monthly Expense Trend Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Expense')
    monthly_trand= plot_to_base64(plt.gcf())

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Amount (INR)', y='Category', data=expense_df)
    plt.title('Expense Summary of analysis')
    plt.xlabel('Amount (INR)')
    plt.ylabel('Category')
    summary_plot = plot_to_base64(plt.gcf())

    return render_template('expense.html', total_expense=total_expense, 
                           expense_by_category=expense_by_category.to_html(), top_expenses_chart=top_expenses_chart, 
                           expense_trends_chart=expense_trends_chart,
                           pie_chart=pie_chart,top_expenses_monthly=top_expenses_monthly,
                           monthly_trand=monthly_trand,summary_plot=summary_plot)
@app.route('/analysis')
def budget_analysis():
    # Perform some basic analysis
    total_expenses = budget_df['Amount (INR)'].sum()
    total_profit_margin = budget_df['Savings'].sum()

    # Plot a bar chart for different categories
    plt.figure(figsize=(10, 6))
    budget_df.groupby('Category')['Amount (INR)'].sum().sort_values().plot(kind='barh')
    plt.title('Total Expenses by Category')
    plt.xlabel('Total Amount (INR)')
    plt.ylabel('Category')
    plt.tight_layout()
    expense_category_chart=plot_to_base64(plt.gcf())

    plt.figure(figsize=(6, 6))
    labels = ['Total Expenses', 'Total Savings']
    values = [total_expenses, total_profit_margin]
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Comparison: Expenses vs Savings')
    plt.tight_layout()
    comparison_pie_chart = plot_to_base64(plt.gcf())
    plt.figure(figsize=(10, 6))


    # Plotting expenses trend
    plt.plot(expense_df['Amount (INR)'], label='Total Expenses', marker='o')

    # Plotting Savings trend
    plt.plot( income_df['Savings'], label='Savings', marker='o')

    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.title('Trend: Expenses vs Savings Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    trands=plot_to_base64(plt.gcf())
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=income_df['amount'], y=expense_df['Amount (INR)'], data=income_df)
    plt.title(' Income vs Expense Analysis')
    plt.xlabel('Amount (INR)')
    plt.ylabel('amount')
    in_chart = plot_to_base64(plt.gcf())

    correlation_matrix = budget_df[['amount', 'Savings', 'Amount (INR)']].corr()
    plt.figure(figsize=(10, 8))
    # Create a heatmap using Seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    # Set the title of the heatmap
    plt.title('Income, Savings, and Expense summary of Analysis')
    sup_chart=plot_to_base64(plt.gcf())

    plt.figure(figsize=(10, 6))
    plt.hist(expense_df['Amount (INR)'], bins=10, alpha=0.5, label='Actual Expenses')

        # Plot grouped histogram graph
    plt.figure(figsize=(10, 6))
    sns.histplot(data=budget_comparison_df, bins=20, alpha=0.5, multiple="stack")
    plt.xlabel('Amount (INR)')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Expenses')
    plt.legend(labels=['Actual', 'Predicted'])
    plt.grid(True)
    plt.tight_layout()
    expenses_predictions_chart = plot_to_base64(plt.gcf())

    Income_Pie_chart_fig, ax = plt.subplots()
    ax.pie([total_actual_income, total_predicted_income], labels=['Actual Income', 'Predicted Income'], autopct='%1.0f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Actual vs Predicted Income')
    Income_Pie_chart = plot_to_base64(Income_Pie_chart_fig)
    
    
    plt.figure(figsize=(10, 6))

    # Plotting predicted expenses
    plt.plot(expense_df.index, expense_df['Amount (INR)'], color="blue", label='Predicted Expenses')

    # Plotting predicted income
    plt.plot(income_df.index, income_df['Predicted_Income'], color="red", label='Predicted Income')

    # Fill between the curves
    plt.fill_between(expense_df.index, expense_df['Amount (INR)'], color="skyblue", alpha=0.2)
    plt.fill_between(income_df.index, income_df['Predicted_Income'], color="salmon", alpha=0.2)

    plt.xlabel('Time')    
    plt.ylabel('Amount (INR)')
    plt.title('Predicted Expenses vs Predicted Income')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    overal_predict=plot_to_base64(plt.gcf())

    return render_template('analysis.html', total_expenses=total_expenses,
                            total_profit_margin=total_profit_margin, 
                            expense_category_chart= expense_category_chart,
                            comparison_pie_chart=comparison_pie_chart,
                            trands=trands,in_chart=in_chart,sup_chart=sup_chart,
                            expenses_predictions_chart=expenses_predictions_chart,
                            Income_Pie_chart=Income_Pie_chart,overal_predict=overal_predict)

if __name__ == '__main__':
    app.run(debug=True)
