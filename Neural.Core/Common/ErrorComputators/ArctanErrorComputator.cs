﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural.Core.Common.ErrorComputators
{
    public class ArctanErrorComputator : IErrorComputator<double>
    {
        public double ComputeError(double[] outputSet, double[] expectedSet)
        {
            if (outputSet.Length != expectedSet.Length)
            {
                throw new ArgumentException($"Lengthes of {nameof(outputSet)} and {nameof(expectedSet)} are not equal");
            }

            double error = outputSet.Select((x, i) => Math.Pow(Math.Atan(x - expectedSet[i]), 2)).Average();
            return error;
        }
    }
}
