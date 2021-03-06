# NeuralNetwork
| **`License`** | **`Language`** | **`AppVeyor`** | **`Travis CI`** | **`Coveralls`** |
|-----------------|---------------------|------------------|-------------------|---------------|
| [![PyPi](https://img.shields.io/pypi/l/Django.svg)](LICENSE) | [![PyPI](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg)](https://github.com/YuriyLisovskiy/NeuralNetwork) | [![Build status](https://ci.appveyor.com/api/projects/status/5akmau97m3tstmxn?svg=true)](https://ci.appveyor.com/project/YuriyLisovskiy/neuralnetwork) | [![Build Status](https://travis-ci.org/YuriyLisovskiy/NeuralNetwork.svg)](https://github.com/YuriyLisovskiy/NeuralNetwork) | [![Coverage Status](https://coveralls.io/repos/github/YuriyLisovskiy/NeuralNetwork/badge.svg)](https://github.com/YuriyLisovskiy/NeuralNetwork) |
## Installation
Linux:
```bash
$ git clone https://github.com/YuriyLisovskiy/NeuralNetwork.git
$ cd NeuralNetwork/
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
Windows:
```bash
$ git clone https://github.com/YuriyLisovskiy/NeuralNetwork.git
$ cd NeuralNetwork/
$ virtualenv venv
$ venv/Scripts/activate
$ pip install -r requirements.txt
```
Run demo:
```bash
$ python runner.py test
```
## Usage
- From `neural_network` package import network:
	```python
	from neural_network.network.net import NeuralNetwork
	```
- Create training data for training neural network, example:
	```python
	training_data = [
	    ([0, 0, 0], 0),
	    ([0, 0, 1], 1),
	    ([0, 1, 0], 0),
	    ([0, 1, 1], 0),
	    ([1, 0, 0], 1),
	    ([1, 0, 1], 1),
	    ([1, 1, 0], 0),
	    ([1, 1, 1], 1)
	]
	```
- Create new neural network using `config/config.py` or custom parameters, example:
	```python
	INPUT_LAYER = [3]
	HIDDEN_LAYERS = [5, 4, 2]
	OUTPUT_LAYER = [1]
	ITERATIONS = 10000
	LEARNING_RATE = 0.007
	```
	```python
	new_net = NeuralNetwork(
	    input_layer=INPUT_LAYER,
	    hidden_layers=HIDDEN_LAYERS,
	    output_layer=OUTPUT_LAYER,
	    learning_rate=LEARNING_RATE,
	    log=False
	)
	```
- Train the network:
	```python
	new_net.train(
	    data=training_data,
	    iterations=ITERATIONS,
	    log=False
	)
	```
- Now network is ready to work, example:
	```python
	def get_prediction(input_data):
	    result = new_net.predict(input_data)
	    return result >= 0.5
	```
	```python
	if __name__ == '__main__':
	    print(get_prediction([0, 1, 0]))
	```
## Author
- **[Yuriy Lisovskiy](https://github.com/YuriyLisovskiy)**
## License
This project is licensed under the BSD-2-Clause License - see the [LICENSE](LICENSE) file for details.
