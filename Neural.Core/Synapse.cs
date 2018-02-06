using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural.Core
{
    public class Synapse<T>
    {
        public Neuron<T> From { get; set; }
        public Neuron<T> To { get; set; }
        public double Weight { get; set; }
    }
}
