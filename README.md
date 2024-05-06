# FinTechLabSummerProject_by Satwik
# Overview
This project is part of the FinTech Lab Summer 2024 Programming Task at Georgia Tech. It involves downloading SEC 10-K filings, analyzing the text using OpenAI's API, and visualizing insights in a simple iOS app. The backend is implemented in Python using Flask, and the frontend is an iOS application developed in Swift.
## Tools and Technologies
1.Python: Used for backend development, including data downloading, processing, and analysis.
2.Flask: A lightweight WSGI web application framework used to create the API that serves processed data to the frontend.
3.Swift: Used for iOS app development to create a user-friendly interface for inputting company tickers and displaying visualizations.
4.Xcode: IDE used for developing the iOS application.
5.Plotly: For generating interactive visualizations of the data analysis.
6.sec_edgar_downloader: Python library used to automate the downloading of SEC filings.
7.OpenAI API: Utilized for performing advanced text analysis on the SEC filings to extract insights.
8.Pandas: For data manipulation and analysis.
9.Matplotlib and Plotly Express: Used for creating visualizations in Python.
# Code Practices
1.Modular Design: Code is organized into modules and functions for clarity and reusability.
2.Error Handling: Robust error handling to manage potential failures in data downloading or processing.
3.Documentation: Comprehensive commenting within the code to explain the functionality and logic, enhancing readability and maintainability.
4.Security Practices: Sensitive information like API keys is recommended to be stored in environment variables or secure vaults, although not directly implemented in the provided scripts.
 5.Version Control: Git is used for version control, with a clear commit history that outlines the development process and modifications.
# Implementation
# Task 1: Data Download and Analysis
1.Data Download: Automated downloading of SEC 10-K filings from 1995 through 2023 for selected companies using sec_edgar_downloader.
2.Text Analysis: Utilized OpenAI's API to analyze the text of the filings and generate insights.
3.Visualization: Insights are visualized using Plotly, showing trends and sentiment analysis over time.
# Task 2: iOS Application
1.User Interface: Developed using UIKit in Swift, where users can enter a company ticker.
2.Data Interaction: The app interacts with the Flask backend via HTTP requests to fetch and display data.
3.Visualization Integration: Utilizes WKWebView to display HTML content generated by Plotly, embedded within the iOS app.
# Deployment
The Flask application should be deployed on a cloud platform like Heroku for public accessibility. Instructions for deployment are included in the deployment section of this README.
The iOS app can be run locally for development and testing purposes and deployed to the App Store for distribution.
# Running the Project
# Backend
1.Navigate to the backend directory.
2.Install dependencies: pip install -r requirements.txt.
3.Run the Flask app: python task1.4.py.
# Frontend
1.Open the Xcode project.
2.Configure the project with your development team for signing.
3.Run the app in the simulator or on a real device.
# Conclusion
This project showcases the integration of complex backend processing with a streamlined frontend application, providing valuable insights into corporate financial filings through advanced text analysis and interactive visualizations.
