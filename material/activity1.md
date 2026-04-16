# Activity 1:

There are 2 parts in this activity and a refresher

---
<details>
<summary><strong>(Refresher): The Standard Machine Learning Pipeline</strong></summary>

<br>

Most supervised learning projects follow a standardized workflow. Regardless of whether you are predicting spam emails or house prices, you will generally follow these 7 steps:

1.  **Load Data:** Get the data into your Python environment (typically using the `pandas` library).
2.  **Explore & Preprocess Data:** Clean the data, handle missing values, and convert text into numbers. 
3.  **Split Data:** Divide the data into a **training set** (to teach the model) and a **testing set** (to evaluate the model on data it has never seen before).
4.  **Choose & Create Model:** Select an appropriate algorithm (e.g., Decision Tree for classification, Linear Regression for regression).
5.  **Train Model:** Fit the model to the **training data**. This is where the mathematical "learning" actually happens.
6.  **Make Predictions:** Use the trained model to guess the outputs for the **testing data**.
7.  **Evaluate Model:** Compare the model's guesses against the *actual* known answers in the testing data using scoring metrics.
</details>

---

## **Part 1: Introduction to Keras**

This lab introduces the basic workflow for building and training a neural network using Keras, a deep learning library in Python. Each step is explained clearly so that beginners can follow along and understand the purpose behind the code.


### **1. Import Required Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
from keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
```

**Explanation:**

- `numpy`: Used for numerical operations, such as arrays and mathematical calculations.
- `matplotlib.pyplot`: Used for plotting graphs to visualize data and training results.
- `pandas`: Used for working with datasets in table (DataFrame) format.
- `keras.Input`: Used to define the input shape of the model.
- `tensorflow.keras.Sequential`: Used to build a neural network model layer by layer.
- `tensorflow.keras.layers.Dense`: Used to create fully connected (dense) layers in the network.
- `sklearn.model_selection.train_test_split`: Used to split the dataset into training and testing sets.

<details>
<summary><strong>Test Your Knowledge: Library Roles</strong></summary>

**Q: Why do we need both `numpy` and `pandas` if they both handle data?**
**A:** `pandas` is excellent for loading, viewing, and manipulating structured tabular data (like spreadsheets with column names). However, neural networks perform mathematical matrix operations under the hood. `numpy` provides the highly optimized numerical array structures that TensorFlow and Keras rely on to perform these calculations efficiently.
</details>


### **2. Load the Datasets**

```python
cereal_data = pd.read_csv("https://github.com/ML-Course-2026/session5/raw/refs/heads/main/material/datasets/cereal.csv")
concrete_data = pd.read_csv("https://github.com/ML-Course-2026/session5/raw/refs/heads/main/material/datasets/concrete.csv")
```

**Explanation:**

- `pd.read_csv(...)` loads CSV (comma-separated values) files from URLs.
- Two datasets are loaded:
  - `cereal_data` contains information about different cereals and their nutrition values.
  - `concrete_data` contains information about the ingredients used in making concrete and the resulting compressive strength.


### **3. Preprocess the Data**

```python
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

concrete_features = concrete_data.drop(columns=['CompressiveStrength'])
concrete_target = concrete_data['CompressiveStrength']
```

**Explanation:**

- For both datasets, we separate the **features** (input variables) from the **target** (the value we want to predict).
- In the cereal dataset:
  - Features are nutritional values.
  - Target is the cereal’s rating.
- In the concrete dataset:
  - Features are ingredient quantities.
  - Target is the compressive strength of the concrete.

<details>
<summary><strong>Concept: Features vs. Target (X and y)</strong></summary>

In machine learning terminology, features are often denoted by a capital `X` (representing a matrix of multiple inputs), and the target is denoted by a lowercase `y` (representing a single column or vector of outputs). 

**Q: Why do we use `.drop(columns=[...])` for the concrete dataset but select specific columns for the cereal dataset?**
**A:** Both approaches achieve the same goal: isolating the inputs from the output. Selecting specific columns is useful when a dataset has many irrelevant columns (like ID or Name). Dropping the target column is faster when *all* other columns in the dataset are intended to be used as features.
</details>


### **4. Split Data into Training and Testing Sets**

```python
cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(
    cereal_features, cereal_target, test_size=0.2, random_state=42)

concrete_X_train, concrete_X_test, concrete_y_train, concrete_y_test = train_test_split(
    concrete_features, concrete_target, test_size=0.2, random_state=42)
```

**Explanation:**

- `train_test_split(...)` splits the data into training and testing sets.
- 80% of the data is used for training, and 20% is used for testing.
- `random_state=42` ensures the split is the same every time the code is run (for reproducibility).

<details>
<summary><strong>Test Your Knowledge: Data Splitting</strong></summary>

**Q: Why do we not use 100% of our data to train the model? Wouldn't that make the model smarter?**
**A:** If a model trains on 100% of the data, it might simply memorize the answers rather than learning the underlying patterns. By reserving 20% as unseen test data, we can evaluate how well the model generalizes to new, real-world data it has never encountered before.
</details>


### **5. Define the Neural Network Model**

```python
model = Sequential()
model.add(Input(shape=(cereal_X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

**Explanation:**

- A **Sequential model** allows us to stack layers one after another.
- `Input(shape=(...))`: Defines the shape of the input data. `cereal_X_train.shape[1]` gives the number of features.
- `Dense(64, activation='relu')`: Adds a dense (fully connected) layer with 64 neurons and ReLU activation function.
- `Dense(32, activation='relu')`: Adds another dense layer with 32 neurons.
- `Dense(1)`: Output layer with one neuron (since this is a regression task predicting a single numeric value).

<details>
<summary><strong>Concept: Architecture Choices</strong></summary>

**Q: What does the number 64 mean in `Dense(64)`?**
**A:** It represents the number of neurons (or nodes) in that specific layer. More neurons allow the network to learn more complex representations, but they also increase the computational cost and the risk of overfitting.

**Q: Why does the final layer have exactly 1 neuron and no activation function?**
**A:** This is a regression task, meaning we are predicting a single continuous number (a cereal rating). One neuron outputs one number. By not specifying an activation function, it defaults to 'linear', allowing the network to output any numerical value (negative or positive) without artificially restricting the range.
</details>


### **6. Compile the Model**

```python
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
```

**Explanation:**

- `optimizer='adam'`: The optimizer adjusts the weights during training. Adam is commonly used and performs well in most cases.
- `loss='mean_squared_error'`: The loss function measures how far predictions are from actual values. MSE is suitable for regression tasks.
- `metrics=['mae']`: Mean Absolute Error is used to monitor performance during training.

<details>
<summary><strong>Test Your Knowledge: Loss vs. Metrics</strong></summary>

**Q: What is the difference between a loss function and a metric?**
**A:** The **loss function** is the mathematical formula the optimizer uses to update the model's weights. The model actively tries to minimize this value during training. A **metric** is calculated strictly for human readability and evaluation. The optimizer ignores the metric when making weight updates.
</details>


### **7. Train the Model**

```python
history_cereal = model.fit(
    cereal_X_train, cereal_y_train,
    epochs=10,
    batch_size=64,
    validation_data=(cereal_X_test, cereal_y_test),
    verbose=1
)
```

**Explanation:**

- `.fit(...)` trains the model on the training data.
- `epochs=10`: The model will go through the entire training data 10 times.
- `batch_size=64`: The model updates weights every 64 samples.
- `validation_data=(...)`: Evaluates model performance on test data during training.
- `verbose=1`: Displays training progress.

<details>
<summary><strong>Concept: Epochs vs. Batches</strong></summary>

**Q: If there are 640 samples in the training data and the batch size is 64, how many times will the model update its weights in a single epoch?**
**A:** 10 times. An epoch represents one full pass over the entire dataset. If the dataset is divided into batches of 64, it takes 10 steps (640 / 64) to complete one epoch. The weights are updated after each batch.
</details>


### **8. Visualize the Training History**

```python
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Explanation:**

- We use `matplotlib.pyplot` to create a line plot.
- The graph shows how the loss (error) changes over time during training.
- Lower validation loss indicates better performance on unseen data.
- Plotting both training and validation loss helps detect overfitting (when the model performs well on training data but poorly on test data).

<details>
<summary><strong>Test Your Knowledge: Graph Interpretation</strong></summary>

**Q: What does it mean if the training loss curve goes down, but the validation loss curve starts going up?**
**A:** This is a classic indicator of overfitting. The model is memorizing the specific data points in the training set (lowering training loss) but losing its ability to generalize to new data (increasing validation loss).
</details>


> [!TIP]
> Here's the [complete code](./src/part1.py)


---

## **Part 2: Early Stopping**

In this part, we improve our training process by using **early stopping**. Early stopping is a technique that monitors model performance on validation data during training. If the model stops improving for several epochs, training will automatically stop. This helps prevent **overfitting**, where the model performs well on training data but poorly on unseen data.


### **1. Import Required Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
```

**Explanation:**

- `EarlyStopping`: A Keras callback that stops training when performance stops improving.
- Other imports are the same as in Part 1 and support data processing, visualization, and model building.


### **2. Load and Preprocess the Dataset**

```python
cereal_data = pd.read_csv("https://github.com/ML-Course-2026/session5/raw/refs/heads/main/material/datasets/cereal.csv")

cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(
    cereal_features, cereal_target, test_size=0.2, random_state=42)
```

**Explanation:**

- The cereal dataset is loaded and split into features (inputs) and target (output).
- The data is then divided into training and testing sets using an 80/20 split.


### **3. Define the Neural Network Model**

```python
model = Sequential()
model.add(Input(shape=(cereal_X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

**Explanation:**

- We use a **Sequential** model with three layers.
- The first layer receives the input shape.
- The second and third layers are hidden layers using ReLU activation.
- The final layer outputs a single value, suitable for a regression task.

<details>
<summary><strong>Concept Reminder: Resetting the Model</strong></summary>

**Q: We already defined this model in Part 1. Why do we need to define it again here?**
**A:** When a model is trained, its weights are updated and saved internally. If we run `.fit()` again on the old model object, it will continue training from where it left off. Redefining the `Sequential()` model initializes a fresh network with new, randomized weights.
</details>


### **4. Compile the Model**

```python
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
```

**Explanation:**

- The model is compiled using the **Adam** optimizer.
- The **mean squared error** is used as the loss function, which is appropriate for regression.
- The **mean absolute error (MAE)** is used as an evaluation metric.


### **5. Define the Early Stopping Callback**

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)
```

**Explanation:**

This callback automatically stops training under certain conditions:

- `monitor='val_loss'`: Tracks validation loss (performance on the test set).
- `patience=3`: If validation loss does not improve for 3 consecutive epochs, training stops.
- `verbose=1`: Displays a message when early stopping is triggered.
- `restore_best_weights=True`: After stopping, the model’s weights are restored to the best point (lowest validation loss).

This helps prevent the model from learning too much from the training data and overfitting.

<details>
<summary><strong>Test Your Knowledge: Patience Parameter</strong></summary>

**Q: Why use `patience=3` instead of `patience=0`?**
**A:** Training curves are rarely perfectly smooth. The validation loss might fluctuate, going up slightly for one epoch before dropping significantly in the next. If `patience` is set to 0, training would stop at the very first minor fluctuation. Adding patience allows the model a brief grace period to recover and find a better minimum.
</details>


### **6. Train the Model with Early Stopping**

```python
history_cereal = model.fit(
    cereal_X_train, cereal_y_train,
    epochs=100,
    batch_size=64,
    validation_data=(cereal_X_test, cereal_y_test),
    callbacks=[early_stopping],
    verbose=1
)
```

**Explanation:**

- We train for up to 100 epochs.
- With early stopping, the model may stop earlier if validation loss doesn't improve.
- `callbacks=[early_stopping]` activates early stopping during training.

<details>
<summary><strong>Concept: High Epoch Limits</strong></summary>

**Q: Why set `epochs=100` if we expect the model to stop early?**
**A:** Setting a high number of maximum epochs combined with Early Stopping guarantees that the model has as much time as it needs to converge. It shifts the burden of deciding "when to stop" from the user guessing a specific epoch number, to an automated, data-driven algorithm.
</details>


### **7. Visualize the Training History**

```python
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Explanation:**

- This graph shows the training and validation loss over each epoch.
- If the lines start to diverge (training loss goes down while validation loss goes up), that can indicate overfitting.
- Early stopping helps stop training before that happens.



### **Does Early Stopping Help Prevent Overfitting?**

Yes. Here’s how early stopping improves model training:

1. **Prevents Overfitting**: Stops training when the model stops improving on validation data.
2. **Improves Generalization**: Helps the model perform better on unseen data by not overfitting to training data.
3. **Saves Time**: Stops unnecessary epochs and reduces training time.

Early stopping is a practical and efficient way to control training and improve model reliability.

> [!TIP]
> Here's the [complete code](./src/part2.py)

----

<details>
<summary><strong>Evaluation and Bonus Challenges</strong></summary>

<br>
Now that you understand the training workflow and early stopping, let's complete the pipeline by evaluating the model and applying your knowledge to a new dataset.

### **1. Interpreting Training Output and Evaluating**

During training, Keras outputs a progress bar with metrics.
`Epoch 10/100  64/64 [==============================] - 0s 2ms/step - loss: 12.34 - mae: 2.50 - val_loss: 14.10 - val_mae: 2.80`

*   `loss` / `mae`: The error on the data the model is currently learning from.
*   `val_loss` / `val_mae`: The error on the unseen validation data. This is the true measure of generalization.

**Task:** Add the following code to the end of your Part 2 script to formally evaluate your best model and see actual predictions. 

```python
# Evaluate the model on the test data
print("\n--- Final Model Evaluation ---")
test_loss, test_mae = model.evaluate(cereal_X_test, cereal_y_test, verbose=0)
print(f"Test MAE: {test_mae:.2f} rating points")

# Verify restore_best_weights=True worked by checking if Test MAE matches 
# the best val_mae seen during training.

# Make a few predictions
sample_predictions = model.predict(cereal_X_test[:5])
print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"Predicted: {sample_predictions[i][0]:.2f} | Actual: {cereal_y_test.iloc[i]:.2f}")
```

### **2. The Concrete Dataset Challenge**

In Part 1, we loaded a `concrete_data` dataset but never used it. 

**Task:** Create a new code cell and build a complete pipeline for the concrete dataset from scratch. 
1. Scale the features (`concrete_X_train` and `concrete_X_test`) using `sklearn.preprocessing.StandardScaler`.
2. Build a `Sequential` model (experiment with layer sizes, e.g., 128 -> 64).
3. Compile the model (use `mse` loss and `mae` metric).
4. Train the model using `EarlyStopping` (set `epochs=200` to ensure early stopping has enough time to trigger).
5. Evaluate the model and plot the loss curves.

**Hint: Scaling Code**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
concrete_X_train_scaled = scaler.fit_transform(concrete_X_train)
concrete_X_test_scaled = scaler.transform(concrete_X_test)
# Use these scaled variables in model.fit() and model.evaluate()
```

</details>