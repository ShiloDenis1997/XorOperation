using Neural.Core;
using Neural.Core.Common.ActivationFunctions;
using Neural.Core.Common.ErrorComputators;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XorOperation
{
    class Program
    {
        private Func<double, Synapse<double>, double> _aggregationFunc;

        static void Main(string[] args)
        {
            List<Neuron<double>> inputLayer = new List<Neuron<double>>();
            List<List<Neuron<double>>> hiddenLayer = new List<List<Neuron<double>>>();
            List<Neuron<double>> outputLayer = new List<Neuron<double>>();
            IActivationFunction<double> activationFunction = new Sigmoid();

            inputLayer.Add(new Neuron<double>(activationFunction, Aggregate));
            inputLayer.Add(new Neuron<double>(activationFunction, Aggregate));

            var firstLayer = new List<Neuron<double>>();
            firstLayer.Add(new Neuron<double>(activationFunction, Aggregate));
            firstLayer.Add(new Neuron<double>(activationFunction, Aggregate));
            hiddenLayer.Add(firstLayer);

            outputLayer.Add(new Neuron<double>(activationFunction, Aggregate));

            inputLayer[0].ConnectTo(hiddenLayer[0][0], 0.45);
            inputLayer[0].ConnectTo(hiddenLayer[0][1], 0.78);

            inputLayer[1].ConnectTo(hiddenLayer[0][0], -0.12);
            inputLayer[1].ConnectTo(hiddenLayer[0][1], 0.13);

            hiddenLayer[0][0].ConnectTo(outputLayer[0], 1.5);
            hiddenLayer[0][1].ConnectTo(outputLayer[0], -2.3);

            IErrorComputator<double> errorComputator = new MseErrorComputator();
            NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, hiddenLayer, outputLayer, errorComputator, 0.7, 0.01);

            List<TrainCase> trainCases = new List<TrainCase>
            {
                new TrainCase{ InputSet = new [] { 0.0, 0.0 }, ExpectedSet = new [] { 0.0 } },
                new TrainCase{ InputSet = new [] { 1.0, 0.0 }, ExpectedSet = new [] { 1.0 } },
                new TrainCase{ InputSet = new [] { 0.0, 1.0 }, ExpectedSet = new [] { 1.0 } },
                new TrainCase{ InputSet = new [] { 1.0, 1.0 }, ExpectedSet = new [] { 0.0 } }
            };

            double newError = trainCases.Max(t => neuralNetwork.TrainOnIteration(t.InputSet, t.ExpectedSet)); ;
            Stopwatch sw = Stopwatch.StartNew();
            double error;
            do
            {
                error = newError;
                newError = trainCases.Max(t => neuralNetwork.TrainOnIteration(t.InputSet, t.ExpectedSet));
                Console.WriteLine($"Error: {newError}");
            } while (Math.Abs(error - newError) > 1e-6);

            sw.Stop();
            Console.WriteLine($"Trained in {sw.ElapsedMilliseconds / 1000.0} seconds");

            while (true)
            {
                double[] input = Console.ReadLine().Split(new[] { ' ' }).Select(s => double.Parse(s)).ToArray();
                neuralNetwork.ComputeOnInputSet(input);
                Console.WriteLine(neuralNetwork.OuputLayer[0].OutputValue);
            }
        }

        public static double Aggregate(double accumulator, Synapse<double> synapse)
        {
            return accumulator + synapse.From.OutputValue * synapse.Weight;
        }

        public class TrainCase
        {
            public double[] InputSet { get; set; }
            public double[] ExpectedSet { get; set; }
        }
    }
}
