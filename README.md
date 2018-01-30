# NeuralNetwork
[![PyPi](https://img.shields.io/pypi/l/Django.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/YuriyLisovskiy/NeuralNetwork)
[![Build Status](https://travis-ci.org/YuriyLisovskiy/NeuralNetwork.svg)](https://github.com/YuriyLisovskiy/NeuralNetwork)
[![Coverage Status](https://coveralls.io/repos/github/YuriyLisovskiy/NeuralNetwork/badge.svg)](https://github.com/YuriyLisovskiy/NeuralNetwork)
### Download and prepare for using
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
### Usage
* From `neural_network` package import network and training:
	```python
	from neural_network.network.net import NeuralNetwork
	from neural_network.learning.training import training
	```
* Create `training_data` for training neural network, example:
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
* Create new neural network using `config/config.py` or custom parameters, example:
	```python
	EPOCHS = 10000
	LEARNING_RATE = 0.007
	LAYERS = [3, 8, 1]
	```
	```python
	new_net = NeuralNetwork(layers=LAYERS, learning_rate=LEARNING_RATE)
	training(neural_net=new_net, training_data=training_data, epochs=EPOCHS)
	```
* Now network is ready to work.
### Author
 * **[Yuriy Lisovskiy](https://github.com/YuriyLisovskiy)**
 ### License
 This project is licensed under the BSD-2-Clause License - see the [LICENSE](LICENSE) file for details.
