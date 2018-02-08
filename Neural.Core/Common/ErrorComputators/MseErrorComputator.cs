using System;
using System.Linq;

namespace Neural.Core.Common.ErrorComputators
{
    public class MseErrorComputator : IErrorComputator<double>
    {
        public double ComputeError(double[] outputSet, double[] expectedSet)
        {
            if (outputSet.Length != expectedSet.Length)
            {
                throw new ArgumentException($"Lengthes of {nameof(outputSet)} and {nameof(expectedSet)} are not equal");
            }

            double error = outputSet.Select((x, i) => Math.Pow(x - expectedSet[i], 2)).Average();
            return error;
        }
    }
}
