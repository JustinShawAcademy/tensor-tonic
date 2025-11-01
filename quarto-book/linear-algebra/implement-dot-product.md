# Implement Dot Product {.unnumbered}

Question: Easy - [29. Implement Dot Product](https://www.tensortonic.com/problems/dot-product)

## Background

## Approach
$$
x \cdot y = \sum_{i=0}^{n-1} x_i \, y_i
$$

Before we compute the formula, we must convert the input's given by the function to NumPy arrays of type `float`

By using `np.asarray`it allows vectorization (no Python loops needed)

```py
import numpy as np
def dot_product(x, y):
    x, y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)
```

After, we can simply use NumPy's efficient C-level dot product operation, `np.dot()` equivalent to 

$$
x_1 y_1 + x_2 y_2 + \dots + x_n y_n
$$

but runs thousands of times faster for large arrays.


```py
import numpy as np
def dot_product(x, y):
    x, y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)
    return np.dot(x, y)
```



![](../images/cat.png)
