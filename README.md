# FashionMNIST
Image classification of clothing made using Tensorflow to understand `tf.Keras`.

## Test & Run

1. Clone this repo using command: `git clone https://github.com/RyanMaugin/FashionMNIST/`.
2. Navigate into `/FashionMNIST` directory using command: `cd FasionMNIST`.
3. Run `nn.py` using command: `python nn.py`.
4. Model should train for 5 epochs which should take around 10 seconds and then open a window with a few visual predictions displayed.

**Note:** The logs in the terminal should also print out more info about the data and training.

## Optimiser Performance Comparison

I tested out different optimisation algorithms to see how long training would take for 5 epochs and what accuracy percentage would be yielded. Here are the results:

| Optimisation                 | Epochs | Accurancy | Training Time |
|------------------------------|--------|-----------|---------------|
| Gradient Descent (𝝰 = 0.01)  | 5      | 84%       | ~10s          |
| Adam                         | 5      | 86.3%     | ~10s          |
| Ada Delta Optimizer          | 5      | 47.2%     | ~10s          |
| Adagrad Optimizer (𝝰 = 0.01) | 5      | 85%       | ~10s          |

The clear winner here is the **Adam** optimisation method which augments classical gradient descent by usign adaptive learning 
rate for each weight. As a result this made the model converge faster than other optimisation algorithms during training. 

## Author

🤖Ryan Maugin   •  🐦@techedryan  •  📬ryanmaugin@icloud.com
