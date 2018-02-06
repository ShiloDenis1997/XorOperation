using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural.Core
{
    public class Neuron<T>
    {
        private Func<T, T> _activationFunc;
        private Func<T, Synapse<T>, T> _aggregationFunc;
        private List<Synapse<T>> _inputs = new List<Synapse<T>>();
        private List<Synapse<T>> _outputs = new List<Synapse<T>>();
        public T OutputValue { get; set; }

        public Neuron(Func<T, T> activationFunc, Func<T, Synapse<T>, T> aggregationFunc)
        {
            _activationFunc = activationFunc;
            _aggregationFunc = aggregationFunc;
        }

        public void ComputeOnIteration()
        {
            T result = _inputs.Aggregate(default(T), (a, s) => a = _aggregationFunc(a, s));
            OutputValue = _activationFunc(result);
        }

        public void ConnectTo(Neuron<T> neuron, double weight)
        {
            var synapse = new Synapse<T> { From = this, To = neuron, Weight = weight };
            _outputs.Add(synapse);
            neuron._inputs.Add(synapse);
        }
    }
}
