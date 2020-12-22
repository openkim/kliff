from kliff import nn
from kliff.analyzers import EnergyForcesRMSE
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.models import NeuralNetwork

# model
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
)

N1 = 10
N2 = 10
model = NeuralNetwork(descriptor)
model.add_layers(
    # first hidden layer
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    # second hidden layer
    nn.Linear(N1, N2),
    nn.Tanh(),
    # output layer
    nn.Linear(N2, 1),
)
model.load("./saved_model.pkl", mode="train")

# dataset
tset = Dataset()
dataset_name = "Si_training_set"
tset.read(dataset_name)
configs = tset.get_configs()

# calculator
calc = CalculatorTorch(model)
calc.create(configs, reuse=True)

# analyzer
analyzer = EnergyForcesRMSE(calc)
analyzer.run(verbose=2, sort="energy")
