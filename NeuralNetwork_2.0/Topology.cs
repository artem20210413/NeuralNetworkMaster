using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_2._0
{


    internal class Topology
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        public double LearningRate { get; set; } // Скорость обучения
        public double Accuracy = 0.0;
        public bool ResetSave = false; //true - delete current file
        public List<int> HiddenLayers { get; }
        public EnumActivationFunction EnumActivationFunction = EnumActivationFunction.Sigmoid;


        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers); // количество нейронов в слое
        }


        public void consoleInfo()
        {
            Console.WriteLine( $"Кількість вхідних нейронов:  {InputCount}");
            string hiddenLayersString = string.Join(", ", HiddenLayers);
            Console.WriteLine( $"Скриті слої та нейрони:  {hiddenLayersString}");
            Console.WriteLine( $"Кількість вихідних нейронов:  {OutputCount}");
            Console.WriteLine( $"Швидкість навчання:  {LearningRate}");
            Console.WriteLine( $"Функція активації:  {Convert.ToString(EnumActivationFunction)}");
        }
    }
}
