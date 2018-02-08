using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural.Core
{
    public class NeuralNetwork <T>
    {
        private List<Neuron<T>> _inputLayer;
        private List<List<Neuron<T>>> _hiddenLayer;
        private List<Neuron<T>> _outputLayer;

        public List<Neuron<T>> OuputLayer => _outputLayer;

        public IErrorComputator<T> ErrorComputator { get; set; }

        public NeuralNetwork(IErrorComputator<T> errorComputator)
        {
            _inputLayer = new List<Neuron<T>>();
            _hiddenLayer = new List<List<Neuron<T>>>();
            _outputLayer = new List<Neuron<T>>();
            ErrorComputator = errorComputator;
        }

        public NeuralNetwork(
            List<Neuron<T>> inputLayer, 
            List<List<Neuron<T>>> hiddenLayer, 
            List<Neuron<T>> outputLayer, 
            IErrorComputator<T> errorComputator)
        {
            _inputLayer = inputLayer;
            _hiddenLayer = hiddenLayer;
            _outputLayer = outputLayer;
            ErrorComputator = errorComputator;
        }

        public void TrainOnIteration(T[] inputSet, T[] expectedSet)
        {
            ComputeOnInputSet(inputSet);
            double error = ErrorComputator.ComputeError(_outputLayer.Select(n => n.OutputValue).ToArray(), expectedSet);
            Console.WriteLine(error);
        }

        private void ComputeOnInputSet(T[] inputSet)
        {
            InitializeInputLayer(inputSet);

            foreach (List<Neuron<T>> layer in _hiddenLayer)
            {
                foreach (Neuron<T> neuron in layer)
                {
                    neuron.ComputeOnIteration();
                }
            }

            foreach (Neuron<T> neuron in _outputLayer)
            {
                neuron.ComputeOnIteration();
            }
        }

        private void InitializeInputLayer(T[] inputSet)
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
