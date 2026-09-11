# Temperature Forecasting using RNN

A deep learning project that uses a **Recurrent Neural Network (RNN)** to forecast future temperature values from historical time-series data.

## 📌 Overview

Temperature forecasting is a time-series prediction problem where historical temperature observations are used to learn patterns and predict future values.

In this project, an **RNN model** is trained on historical temperature data. The model learns sequential patterns in the data and uses previous observations to predict the next temperature value.

## 🎯 Objectives

* Understand time-series forecasting using deep learning.
* Preprocess and normalize historical temperature data.
* Create sequential input data suitable for an RNN.
* Train an RNN model for temperature prediction.
* Evaluate the model's forecasting performance.
* Visualize actual and predicted temperatures.

## 🧠 How RNN is Used

An RNN is suitable for this problem because temperature data is sequential.

For example:

```text
Previous temperatures
        ↓
[25.1, 25.4, 26.0, 26.3, 26.7]
        ↓
       RNN
        ↓
Predicted temperature
        ↓
      27.0
```

The model uses information from previous time steps to learn patterns and generate a prediction for the next time step.

## 🔄 Project Workflow

```text
Historical Temperature Data
            ↓
       Data Cleaning
            ↓
     Data Normalization
            ↓
    Create Time Sequences
            ↓
       Train/Test Split
            ↓
       RNN Model
            ↓
         Training
            ↓
        Prediction
            ↓
 Model Evaluation & Visualization
```

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* Recurrent Neural Network (RNN)

## 📂 Project Structure

```text
temperature-forecasting-rnn/
│
├── data/
│   └── temperature_data.csv
│
├── notebooks/
│   └── temperature_forecasting.ipynb
│
├── src/
│   └── model.py
│
├── results/
│   └── predictions.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/komaldev-eng/temperature-forecasting-rnn.git
```

Move into the project directory:

```bash
cd temperature-forecasting-rnn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Run the Python script:

```bash
python src/model.py
```

Or open the Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebook inside the `notebooks` directory.

## 📊 Model Evaluation

The model can be evaluated using regression metrics such as:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**

The project also compares the **actual temperature values** with the **predicted values** using visualization.

## 📈 Results

The trained RNN learns temporal patterns from historical temperature observations and generates predictions for future time steps.

A plot of actual vs. predicted temperature values can be added here:

```text
Actual Temperature  ─────────────
Predicted Temperature ───────────
```

> Add your final result graph/screenshot here after training the model.

## 💡 Key Learnings

Through this project, I learned:

* How time-series data is structured.
* Why sequential data requires special preprocessing.
* How to create sliding-window sequences.
* How RNNs process sequential information.
* How normalization affects neural-network training.
* How to evaluate regression models.
* How to visualize forecasting results.

## 🚀 Future Improvements

* Experiment with **LSTM** and **GRU** models.
* Use larger and more diverse weather datasets.
* Add multiple weather features such as humidity, pressure, and wind speed.
* Perform multi-step temperature forecasting.
* Deploy the model as a web application or API.

## 👩‍💻 Author

**Komal Agarwal**

B.Tech – Data Science

GitHub: [komaldev-eng](https://github.com/komaldev-eng)
