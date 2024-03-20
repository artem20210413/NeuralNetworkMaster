using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_2._0
{
    public enum EnumActivationFunction
    {
        Sigmoid,        // Логистическая (Сигмоида или Гладкая ступенька)
        ReLU,           // Линейный выпрямитель
        Tanh,           // Гиперболического тангенса
        SoftPlus,       // SoftPlus                                                 оч?куваний: 1,15, фактичний: не число
        Gaussian,       // Гауссова
        Identical,      // Тождественное
        LeakyReLU,      // Линейный выпрямитель с «утечкой»
        Sin,            // Синусоида
        Softmax,            // Softmax
    }

    public class ActivationFunction
    {
        private EnumActivationFunction type { get; set; }

        public ActivationFunction(EnumActivationFunction type){
            this.type = type;
        }


        public double Activation(double x)
        {

            switch (this.type)
            {
                case EnumActivationFunction.Sigmoid:
                    return Sigmoid(x);
                case EnumActivationFunction.ReLU:
                    return ReLU(x);
                case EnumActivationFunction.Tanh:
                    return Tanh(x);
                case EnumActivationFunction.SoftPlus:
                    return SoftPlus(x);
                case EnumActivationFunction.Gaussian:
                    return Gaussian(x);
                case EnumActivationFunction.Identical:
                    return Identical(x);
                case EnumActivationFunction.LeakyReLU:
                    return LeakyReLU(x);
                case EnumActivationFunction.Sin:
                    return Sin(x);
                case EnumActivationFunction.Softmax:
                    return Softmax(x);

                default: return Sigmoid(x);
            }

        }

        public double ActivationDx(double x)
        {
            switch (this.type)
            {
                case EnumActivationFunction.Sigmoid:
                    return SigmoidDx(x);
                case EnumActivationFunction.ReLU:
                    return ReLUDx(x);
                case EnumActivationFunction.Tanh:
                    return TanhDx(x);
                case EnumActivationFunction.SoftPlus:
                    return SoftPlusDx(x);
                case EnumActivationFunction.Gaussian:
                    return GaussianDx(x);
                case EnumActivationFunction.Identical:
                    return IdenticalDx(x);
                case EnumActivationFunction.LeakyReLU:
                    return LeakyReLUDx(x);
                case EnumActivationFunction.Sin:
                    return SinDx(x);
                case EnumActivationFunction.Softmax:
                    return SoftmaxDx(x);

                default: return SigmoidDx(x);
            }
        }


        private double Sigmoid(double x) // Логистическая (Сигмоида или Гладкая ступенька)
        {
            var result = 1.0 / (1.0 + Math.Exp(-x));
            return result;
        }

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid * (1 - sigmoid);

            return result;
        }

        private double ReLU(double x)   // Линейный выпрямитель
        {
            return Math.Max(0, x);
        }

        private double ReLUDx(double x)
        {
            return (x >= 0) ? 1 : 0;
        }

        private double Tanh(double x) // Гиперболического тангенса
        {
            return Math.Tanh(x);
        }

        private double TanhDx(double x)    
        {
            double tanhX = Math.Tanh(x);
            return 1 - tanhX * tanhX;
        }

        private double SoftPlus(double x) // SoftPlus
        {
            return Math.Log(1 + Math.Exp(x));
        }

        private double SoftPlusDx(double x)
        {
            double expX = Math.Exp(x);
            return 1 / (1 + Math.Exp(-x));
        }

        private double Gaussian(double x) // Гауссова
        {
            return Math.Exp(-Math.Pow(x,2));
        }

        private double GaussianDx(double x)
        {
            double gaussianX = Gaussian(x);
            return -2 * x * gaussianX;
        }


        private double Identical(double x) // Тождественное
        {
            return x;
        }

        private double IdenticalDx(double x)
        {
            return 1;
        }

        private double LeakyReLU(double x) // Линейный выпрямитель с «утечкой»
        {
            return (x >= 0) ? x : x * 0.01;
        }

        private double LeakyReLUDx(double x)
        {
            return (x >= 0) ? 1 : x * 0.01;
        }

        private double Sin(double x) // Синусоида
        {
            return Math.Sin(x);
        }

        private double SinDx(double x)
        {
            return Math.Cos(x);
        }

        private double Softmax(double x)
        {
            // Применение Softmax операции к значению x
            double expX = Math.Exp(x);
            double softmax = expX / (1 + expX); // Здесь используется формула для Softmax с одним входом
            return softmax;
        }

        private double SoftmaxDx(double x)
        {
            // Вычисление производной функции Softmax по входному значению x
            double softmax = Softmax(x);
            double softmaxDx = softmax * (1 - softmax);
            return softmaxDx;
        }


        //interface IActivationMethod
        //{
        //    double Activate(double x);
        //    double ActivateDx(double x);
        //}
        //private class LeakyReLU : IActivationMethod
        //{
        //    private double alpha;

        //    public LeakyReLU(double alpha)
        //    {
        //        this.alpha = alpha;
        //    }

        //    public double Activate(double x)
        //    {
        //        return (x >= 0) ? x : alpha * x;
        //    }

        //    public double ActivateDx(double x)
        //    {
        //        return (x >= 0) ? 1 : alpha;
        //    }
        //}

    }



}
