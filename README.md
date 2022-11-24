Completed the following 2 tasks as part of Shack Labs DS Internship Assignment: 

<ol>
  <li>House Price Prediction</li>
  <li>Matching of Amazon and Flipkart Products</li>
</ol>

## Libraries & Frameworks Used

<h3>For Task 1: House Price Prediction</h3>
<ul type='a'>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Scikit-learn</li>
  <li>Scipy</li>
  <li>XGBoost</li>
  <li>LightGBM</li>
  <li>CatBoost</li>
</ul>

<h3>For Task 2: Product Matching</h3>
<ul type='a'>
  <li>PyTorch</li>
  <li>Sentence Transformers</li>
  <li>Pandas</li>
</ul>

## Machine Learning Models Performance Comparison for Task 1: House Price Prediction

<table>
  <tr>
    <th>Model</th>
    <th>R2 Score (%)</th>
  </tr>
  <tr>
    <td>Extra Trees Regressor</td>
    <td>81.64%</td>
  </tr>
  <tr>
    <td>Cat Boost Regressor</td>
    <td>81.39%</td>
  </tr>
  <tr>
    <td>Gradient Boosting Regressor</td>
    <td>78.96%</td>
  </tr>
  <tr>
    <td>XG Boost Regressor</td>
    <td>72.87%</td>
  </tr>
  <tr>
    <td>Random Forest Regressor</td>
    <td>75.49%</td>
  </tr>
  <tr>
    <td>Decision Tree Regressor</td>
    <td>63.80%</td>
  </tr>
  <tr>
    <td>K Nearest Neighbors Regressor</td>
    <td>57.61%</td>
  </tr>
  <tr>
    <td>Support Vector Regressor</td>
    <td>59.28%</td>
  </tr>
  <tr>
    <td>Light Gradient Boosting Regressor</td>
    <td>74.72%</td>
  </tr>
  <tr>
    <td>Bagging Regressor</td>
    <td>74.82%</td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>64.39%</td>
  </tr>
  <tr>
    <td>Multi-Layer Perceptron Regressor</td>
    <td>51.15%</td>
  </tr>
  <tr>
    <td>Histogram Gradient Boosting Regressor</td>
    <td>78.42%</td>
  </tr>
</table>

## Installation of Sentence Transformers for Task 2: Product Matching

We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

**Install with pip**

Install the *sentence-transformers* with `pip`:

```
pip install -U sentence-transformers
```

**Install with conda**

You can install the *sentence-transformers* with `conda`:

```
conda install -c conda-forge sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:

````
pip install -e .
```` 

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

  
 
