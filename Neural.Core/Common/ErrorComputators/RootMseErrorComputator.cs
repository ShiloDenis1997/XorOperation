using System;

namespace Neural.Core.Common.ErrorComputators
{
    public class RootMseErrorComputator : IErrorComputator<double>
    {
        private IErrorComputator<double> _mseErrorComputator;

        public RootMseErrorComputator()
        {
            _mseErrorComputator = new MseErrorComputator();
        }

        public double ComputeError(double[] outputSet, double[] expectedSet)
        {
            return Math.Sqrt(_mseErrorComputator.ComputeError(outputSet, expectedSet));
        }
    }
}
