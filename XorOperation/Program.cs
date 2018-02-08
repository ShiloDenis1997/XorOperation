using Neural.Core;
using Neural.Core.Common.ErrorComputators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XorOperation
{
    class Program
    {
        private Func<double, double> _activationFunc;
        private Func<double, Synapse<double>, double> _aggregationFunc;

        static void Main(string[] args)
        {
            List<Neuron<double>> inputLayer = new List<Neuron<double>>();
            List<List<Neuron<double>>> hiddenLayer = new List<List<Neuron<double>>>();
            List<Neuron<double>> outputLayer = new List<Neuron<double>>();

            inputLayer.Add(new Neuron<double>(Activate, Aggregate));
            inputLayer.Add(new Neuron<double>(Activate, Aggregate));

            var firstLayer = new List<Neuron<double>>();
            firstLayer.Add(new Neuron<double>(Activate, Aggregate));
            firstLayer.Add(new Neuron<double>(Activate, Aggregate));
            hiddenLayer.Add(firstLayer);

            outputLayer.Add(new Neuron<double>(Activate, Aggregate));

            inputLayer[0].ConnectTo(hiddenLayer[0][0], 0.45);
            inputLayer[0].ConnectTo(hiddenLayer[0][1], 0.78);

            inputLayer[1].ConnectTo(hiddenLayer[0][0], -0.12);
            inputLayer[1].ConnectTo(hiddenLayer[0][1], 0.13);

            hiddenLayer[0][0].ConnectTo(outputLayer[0], 1.5);
            hiddenLayer[0][1].ConnectTo(outputLayer[0], -2.3);

            IErrorComputator<double> errorComputator = new MseErrorComputator();
            NeuralNetwork<double> neuralNetwork = new NeuralNetwork<double>(inputLayer, hiddenLayer, outputLayer, errorComputator);
            neuralNetwork.TrainOnIteration(new[] { 1.0, 0.0 }, new[] { 1.0 });

            Console.WriteLine(neuralNetwork.OuputLayer[0].OutputValue);
        }

        public static double Aggregate(double accumulator, Synapse<double> synapse)
        {
            return accumulator + synapse.From.OutputValue * synapse.Weight;
        }

        public static double Activate(double value)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -value));
        }
    }
}
