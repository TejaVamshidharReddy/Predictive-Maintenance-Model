# Predictive Maintenance Model

## Overview

This project implements a machine learning-based predictive maintenance system that analyzes sensor data and maintenance logs to predict equipment failures before they occur. By leveraging historical data patterns, the model helps reduce downtime, optimize maintenance schedules, and extend equipment lifespan. The solution uses classification algorithms to identify patterns that indicate potential equipment failures, enabling proactive maintenance strategies.

## Skills

- Machine Learning
- Data Analysis
- Feature Engineering
- Model Development & Evaluation
- Data Preprocessing
- Statistical Analysis
- Python Programming
- Data Visualization

## Technologies

- **Python**: Core programming language
- **scikit-learn**: Machine learning algorithms and model evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **Jupyter Notebook**: Interactive development and analysis

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/TejaVamshidharReddy/Predictive-Maintenance-Model.git
cd Predictive-Maintenance-Model
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the src directory:
```bash
cd src
```

2. Run the predictive maintenance model:
```bash
python maintenance_model.py
```

3. For exploratory analysis, open the Jupyter notebooks:
```bash
jupyter notebook
```
Then navigate to the `notebooks/` directory and open the desired notebook.

## Input/Output Example

### Input
The model accepts sensor data and maintenance logs with features such as:
- Temperature readings
- Vibration measurements
- Pressure levels
- Operational hours
- Previous maintenance records
- Equipment age

Example input format (CSV):
```
temperature,vibration,pressure,hours_operated,days_since_maintenance,failure
75.5,2.3,45.2,120,15,0
82.1,3.7,48.9,145,28,0
95.3,5.2,52.1,180,45,1
```

### Output
The model outputs:
- Failure probability for each equipment unit
- Classification (maintenance required: yes/no)
- Feature importance scores
- Model performance metrics (accuracy, precision, recall, F1-score)

Example output:
```
Equipment ID: EQ-001
Failure Probability: 0.78
Maintenance Required: YES
Recommended Action: Schedule maintenance within 7 days

Model Accuracy: 92.5%
Precision: 0.89
Recall: 0.91
F1-Score: 0.90
```

## Business Impact

- **Cost Reduction**: Reduce unplanned downtime by up to 50% through early failure detection
- **Optimized Maintenance**: Shift from reactive to proactive maintenance scheduling, reducing unnecessary preventive maintenance by 30%
- **Extended Equipment Life**: Prevent catastrophic failures that can permanently damage equipment
- **Improved Safety**: Identify potential hazards before they lead to accidents
- **Resource Optimization**: Better allocation of maintenance personnel and spare parts inventory
- **Production Continuity**: Minimize production disruptions and maintain consistent output
- **ROI**: Typical ROI of 10:1 through reduced downtime costs and optimized maintenance operations
