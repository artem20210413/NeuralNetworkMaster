using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_2._0
{


    internal class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        public Topology topology { get; private set; }
        public ActivationFunction ActivationFunction { get; private set; }
        Random rnd = new Random();

        public Neuron(int inputCount, NeuronType type, Topology topology)
        {
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();
            this.topology = topology;
            this.ActivationFunction = new ActivationFunction(this.topology.EnumActivationFunction);

            InitWeightsRandomValue(inputCount);
        }

        private void InitWeightsRandomValue(int inputCount)
        {

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input) 
                    Weights.Add(1);
                else
                {
                    Weights.Add(this.HeInitializationInitWeightsRandomValue());
                    //Weights.Add(rnd.NextDouble());
                    //Weights.Add((0.001 - 0.0001) * rnd.NextDouble() + 0.0001);
                }

                Inputs.Add(0);
            }
        }
        private double HeInitializationInitWeightsRandomValue()
        {
            double scale = this.Inputs.Count != 0
                       ? Math.Sqrt(2.0 / this.Inputs.Count)
                       : 1;
             
            return (2 * rnd.NextDouble() - 1) * scale;
            /**
             * Инициализация He (He Initialization) - это метод инициализации весов нейронной сети, который был предложен Каимингом Хе и адаптирован для работы с функцией активации ReLU (Rectified Linear Unit). Этот метод позволяет эффективно начать обучение сети, чтобы избежать проблемы затухающих или взрывающихся градиентов.
             */
        }



        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                 sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input) 
                Output = ActivationFunction.Activation(sum);
            else 
                Output = sum;

                return Output;
        }

        public void Learn(double error, double learningRate)
        {

            if (NeuronType == NeuronType.Input ) return;

            Delta = error * ActivationFunction.ActivationDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeigth = weight - input * Delta * learningRate;
                Weights[i] = newWeigth;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
