using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural.Core
{
    public class NeuralNetwork
    {
        private List<Neuron<double>> _inputLayer;
        private List<List<Neuron<double>>> _hiddenLayer;
        private List<Neuron<double>> _outputLayer;
        private double _learningRate;
        private double _moment;

        public List<Neuron<double>> OuputLayer => _outputLayer;

        public IErrorComputator<double> ErrorComputator { get; set; }

        public NeuralNetwork(IErrorComputator<double> errorComputator, double learningRate, double moment)
        {
            _inputLayer = new List<Neuron<double>>();
            _hiddenLayer = new List<List<Neuron<double>>>();
            _outputLayer = new List<Neuron<double>>();
            _learningRate = learningRate;
            _moment = moment;
            ErrorComputator = errorComputator;
        }

        public NeuralNetwork(
            List<Neuron<double>> inputLayer, 
            List<List<Neuron<double>>> hiddenLayer, 
            List<Neuron<double>> outputLayer, 
            IErrorComputator<double> errorComputator,
            double learningRate, 
            double moment)
        {
            _inputLayer = inputLayer;
            _hiddenLayer = hiddenLayer;
            _outputLayer = outputLayer;
            _learningRate = learningRate;
            _moment = moment;
            ErrorComputator = errorComputator;
        }

        public double TrainOnIteration(double[] inputSet, double[] expectedSet)
        {
            ComputeOnInputSet(inputSet);
            Console.WriteLine($"{inputSet[0]: 0.##} xor {inputSet[1]: 0.##} = {OuputLayer[0].OutputValue}");
            double error = ErrorComputator.ComputeError(_outputLayer.Select(n => n.OutputValue).ToArray(), expectedSet);

            TrainOuputLayer(expectedSet);
            TrainHiddenLayer();
            TrainInputLayer();
            return error;
        }

        private void TrainOuputLayer(double[] expectedSet)
        {
            for (int i = 0; i < expectedSet.Length; i++)
            {
                CountDeltaForOutputNeuron(_outputLayer[i], expectedSet[i]);
            }
        }

        private void TrainInputLayer()
        {
            foreach (var neuron in _inputLayer)
            {
                RecalculateWeights(neuron);
            }
        }

        private void RecalculateWeights(Neuron<double> neuron)
        {
            foreach (var synapse in neuron.Outputs)
            {
                double gradient = synapse.To.Delta * neuron.OutputValue;
                synapse.DeltaWeight = _learningRate * gradient + _moment * synapse.DeltaWeight;
                synapse.Weight += synapse.DeltaWeight;
            }
        }

        private void TrainHiddenLayer()
        {
            for (int i = _hiddenLayer.Count - 1; i >= 0; i--)
            {
                foreach (var neuron in _hiddenLayer[i])
                {
                    CountDeltaForHiddenNeuron(neuron);
                    RecalculateWeights(neuron);
                }
            }
        }

        private void CountDeltaForOutputNeuron(Neuron<double> neuron, double idealOutput)
        {
            neuron.Delta = (idealOutput - neuron.OutputValue) * neuron.ActivationFunction.Derivate(neuron.InputValue);
        }

        private void CountDeltaForHiddenNeuron(Neuron<double> neuron)
        {
            neuron.Delta = neuron.ActivationFunction.Derivate(neuron.InputValue) 
                * neuron.Outputs.Sum(s => s.Weight * s.To.Delta);
        }

        public void ComputeOnInputSet(double[] inputSet)
        {
            InitializeInputLayer(inputSet);

            foreach (List<Neuron<double>> layer in _hiddenLayer)
            {
                foreach (Neuron<double> neuron in layer)
                {
                    neuron.ComputeOnIteration();
                }
            }

            foreach (Neuron<double> neuron in _outputLayer)
            {
                neuron.ComputeOnIteration();
            }
        }

        private void InitializeInputLayer(double[] inputSet)
        {
            if (inputSet.Length != _inputLayer.Count)
            {
                throw new ArgumentException($"number of input value is {inputSet.Length}, but {_inputLayer.Count} expected");
            }

            for (int i = 0; i < _inputLayer.Count; i++)
            {
                _inputLayer[i].OutputValue = inputSet[i];
            }
        }
    }
}
