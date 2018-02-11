using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural.Core
{
    public class Neuron<T>
    {
        private IActivationFunction<T> _activationFunc;
        private Func<T, Synapse<T>, T> _aggregationFunc;
        private List<Synapse<T>> _inputs = new List<Synapse<T>>();
        private List<Synapse<T>> _outputs = new List<Synapse<T>>();
        private Synapse<T> _biasSynapse;

        public T OutputValue { get; set; }
        public T Delta { get; set; }
        public T InputValue { get; private set; }

        public IActivationFunction<T> ActivationFunction => _activationFunc;

        public List<Synapse<T>> Inputs => _inputs;

        public List<Synapse<T>> Outputs => _outputs;

        public Neuron(IActivationFunction<T> activationFunc, Func<T, Synapse<T>, T> aggregationFunc, Synapse<T> biasSynapse = null)
        {
            _activationFunc = activationFunc;
            _aggregationFunc = aggregationFunc;
            _biasSynapse = biasSynapse;
        }

        public void ComputeOnIteration()
        {
            T result = Inputs.Aggregate(default(T), (a, s) => a = _aggregationFunc(a, s));
            if (_biasSynapse != null)
            {
                result = _aggregationFunc(result, _biasSynapse);
            }
            InputValue = result;
            OutputValue = _activationFunc.Activate(result);
        }

        public void ConnectTo(Neuron<T> neuron, double weight)
        {
            var synapse = new Synapse<T> { From = this, To = neuron, Weight = weight };
            Outputs.Add(synapse);
            neuron.Inputs.Add(synapse);
        }
    }
}
