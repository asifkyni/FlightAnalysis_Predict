# Fairness Mitigation with Flights Dataset

This project demonstrates how to perform **fairness mitigation** on a flight dataset using **Fairlearn**. The goal is to mitigate bias in machine learning models, specifically focusing on ensuring **demographic parity** for sensitive features such as the `carrier` column.

## Project Overview

The project includes the following tasks:
1. **Data Preparation**: Load the flight dataset, clean the data, and prepare features and target variables.
2. **Fairness Mitigation**: Apply fairness mitigation techniques using the `ExponentiatedGradient` algorithm from the **Fairlearn** library. This method ensures **demographic parity** between different groups defined by the sensitive feature (e.g., `carrier`).
3. **Model Evaluation**: Assess model fairness and performance using **Demographic Parity Difference** and **Mean Absolute Error (MAE)**.

The focus of the project is to evaluate and mitigate bias in predictive models while ensuring fairness across different demographic groups.

## Getting Started

### Prerequisites

You will need the following Python libraries installed:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **scikit-learn**: For machine learning models and metrics.
- **Fairlearn**: For fairness-aware machine learning algorithms.
  
You can install all dependencies using `pip`:

```bash
pip install -r requirements.txt
