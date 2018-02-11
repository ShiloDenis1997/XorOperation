using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural.Core.Common.ActivationFunctions
{
    public class Sigmoid : IActivationFunction<double>
    {
        public double Activate(double value)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -value));
        }

        public double Derivate(double value)
        {
            double result = Activate(value);
            result = (1 - result) * result;
            return result;
        }
    }
}
