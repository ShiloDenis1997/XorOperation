using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural.Core
{
    public interface IActivationFunction<T>
    {
        T Activate(T value);
        T Derivate(T value);
    }
}
