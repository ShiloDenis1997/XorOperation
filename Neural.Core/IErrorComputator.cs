using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural.Core
{
    public interface IErrorComputator<T>
    {
        double ComputeError(T[] outputSet, T[] expectedSet);
    }
}
