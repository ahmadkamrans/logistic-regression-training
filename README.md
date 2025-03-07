# Logistic Regression Training

### Overview
- Popular Home Features example for basic understanding of the Logistic Regression
- After that, we will move forward to [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris) 

### Single Feature: Linear Regression using Model Function
- Currently, if you take an example of housing dataset: 
    | Size (1000 sqft) | Price (1000s of dollars) |
    |------------------|--------------------------|
    | 1.0              | 300                      |
    | 2.0              | 500                      |

- Implemented the logistic regression by specifically defining the model function. Instead of using the **scikit_learn**. As, for learning purposes to have deeper knowledge of how the things work at the underlying level.
- Used **NumPy**, and **Matplotlib**

- A basic overview is that made separate **np.array**'s for the x_train(Size of house), and y_train(Price of house)

- Then, printed the shape of both to see the dimension of the data

- Simply used the scatter plot on base dataset values:
  Here size of the house was on x-axis, with price of house on the y-axis
  ![alt text](images/plotDataPoints.png)

- Defined the function **compute_model_output(x_train, w, b)** in which the model function was there for the predicting the price based on **x_train**:
  ```python
  f_wb[i] = weight * x[i] + bias
  ```

- Now the function **compute_model_output(x_train, w, b)** will plot the predicted values (prices) on the already placed data points on Scatter plot:
  ![alt text](images/modelFuncPredictedValues.png)

  like if we predict the price for the 1200 square foot house:
  ```python
    x_i = 1.2
    predict = w * x_i + b
  ```

  the predicted price of the house came: **330.0 $**

### Single Feature: Linear Regression Model

- using the same dataset, but this time prediction is done through the Logisitc Regression Model 

- In model function we were assuming the weight, and the bias. But here we are using
  ```python
  model.fit(x_train, y_train)
  ```
  fit basically adjusts the weight, and bias by learning **y = mx+b**

- The prediction is done through:
  ```python
  predicted_price = model.predict([[1.2]])
  ```

- Now, visualizing the base data points in dataset, the predicted values by the model:

![alt text](images/LogisticRegiression2.png)

- Calculating the cost which is between the actual data point, and the predicted point. (i.e: point is price)

```python
cost = 0.5*(y_pred[i] - y_actual[i]) ** 2
```

- Resulted Cost:


| Data Point | x (Size in 1000 sqft) | Actual y (Price in $1000s) | Predicted y (Price in $1000s) | Cost   |
|------------|-----------------------|----------------------------|-------------------------------|--------|
| 1          | 1.0                   | 319.87                     | 310.40                        | 44.87  |
| 2          | 2.0                   | 494.47                     | 513.42                        | 179.47 |
| 3          | 3.0                   | 725.91                     | 716.43                        | 44.87  |

### Single Feature: Gradient Discent from scratch for Linear Regression
- Here, we optimized the weight, and bais 

- Gradient Descent minimizes the **Mean Squared Error (MSE)**:
  As for every iteration first gradient is calculated which adjusts the weight, and the bias
  ```pyhton
  f_wb = w * x[i] + b 
        error = (f_wb - y[i])  
        dj_dw += error * x[i]  
        dj_db += error
  ```
  Here f_wb is the predicted data point, and y[i] is the actual data point 
  dj_dw is the partial derivative with respect to weight
  dj_db is the partial derivative with respect to bias

- Then, the cost is calculated the distance between the actual, and predicted data point
  ```python
  f_wb = w * x[i] + b  
        cost_i = (f_wb - y[i]) ** 2 
        total_cost += cost_i
  ```

- At last the gradient descent is performed to optimize the weight, and the bias
  ```python
  w = w - alpha * dj_dw
  b = b - alpha * dj_db
  ```

- These were the intial parameters:

  w_init = 0  
  b_init = 0  
  alpha = 0.1  
  num_iters = 100  

![alt text](images/image.png)
![alt text](images/image%20copy.png)

### Multiple Feature: Multiple Linear Regression

- First we only had one feature like the price. Now we have multiple features columns.

![alt text](images/Multiple%20Feature.png)

So, now it will not be previous prediction formula: 
```python 
f_wb[i] = weight * x[i] + bias 
```
It wil be updated to include all features:
```python 
f_wb[i] = weight_1 * x_1[i] + weight_2 * x_2 + weight_3 * x_3 + weight_n * x_n + bias 
```

So, as we have list of weights, and features (x) make list of it:
```python
w = [w1,w2,w3,wn]
x = [x1,x2,x3,xn]
```

it will be a dot product:
```python
f_wb = w . x + b
```

- So, now a simple base code was that we implement in that compute the prediction by a for loop. 
- Instead in vectorization we will use **numpy library**:
```python
f = np.dot(w,x) + b
```  
### Dataset

[Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)