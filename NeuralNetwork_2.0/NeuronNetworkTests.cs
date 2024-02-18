using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_2._0
{
    public class NeuronNetworkTests
    {
        static public void FeedForwardTest()
        {
            //var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            //var inputs = new double[,]
            //{
            //    // Результат - Пациент болен - 1
            //    //             Пациент Здоров - 0

            //    // Неправильная температура T
            //    // Хороший возраст A
            //    // Курит S
            //    // Правильно питается F
            //    //T  A  S  F
            //    { 0, 0, 0, 0 },
            //    { 0, 0, 0, 1 },
            //    { 0, 0, 1, 0 },
            //    { 0, 0, 1, 1 },
            //    { 0, 1, 0, 0 },
            //    { 0, 1, 0, 1 },
            //    { 0, 1, 1, 0 },
            //    { 0, 1, 1, 1 },
            //    { 1, 0, 0, 0 },
            //    { 1, 0, 0, 1 },
            //    { 1, 0, 1, 0 },
            //    { 1, 0, 1, 1 },
            //    { 1, 1, 0, 0 },
            //    { 1, 1, 0, 1 },
            //    { 1, 1, 1, 0 },
            //    { 1, 1, 1, 1 }
            //};

            List<double> outputs = new List<double> { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            //List<List<double>> outputs = new List<List<double>>
            //{
            //    new List<double>  { 0 },
            //    new List<double>  { 0 },
            //    new List<double>  { 1 },
            //    new List<double>  { 0 },
            //    new List<double>  { 0 },
            //    new List<double>  { 0 },
            //    new List<double>  { 1 },
            //    new List<double>  { 0 },
            //    new List<double>  { 1 },
            //    new List<double>  { 1 },
            //    new List<double>  { 1 },
            //    new List<double>  { 1 },
            //    new List<double>  { 1 },
            //    new List<double>  { 0 },
            //    new List<double>  { 1 },
            //    new List<double>  { 1 }
            //};

            List<List<double>> inputs = new List<List<double>>
            {
                new List<double>  { 0, 0, 0, 0 },
                new List<double>  { 0, 0, 0, 1 },
                new List<double>  { 0, 0, 1, 0 },
                new List<double>  { 0, 0, 1, 1 },
                new List<double>  { 0, 1, 0, 0 },
                new List<double>  { 0, 1, 0, 1 },
                new List<double>  { 0, 1, 1, 0 },
                new List<double>  { 0, 1, 1, 1 },
                new List<double>  { 1, 0, 0, 0 },
                new List<double>  { 1, 0, 0, 1 },
                new List<double>  { 1, 0, 1, 0 },
                new List<double>  { 1, 0, 1, 1 },
                new List<double>  { 1, 1, 0, 0 },
                new List<double>  { 1, 1, 0, 1 },
                new List<double>  { 1, 1, 1, 0 },
                new List<double>  { 1, 1, 1, 1 }
            };




            //NeuralNetworkDataLoader neuralNetworkDataLoader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSetOutput_1.xlsx");
            //var inputsList = neuralNetworkDataLoader.getInputs();
            //var outputsList = neuralNetworkDataLoader.getOutputs();

            //double[,] inputs = neuralNetworkDataLoader.ProcessListOfLists(inputsList);
            //double[] outputs = neuralNetworkDataLoader.ProcessListOfLists(outputsList);


            //neuralNetworkDataLoader.countColumInput,
            //neuralNetworkDataLoader.countColumOutput,

            Topology topology = new Topology(
                4,
                1,
                0.1,
                3
                );
            topology.EnumActivationFunction = EnumActivationFunction.Sigmoid;

            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            double difference = neuralNetwork.Learn(outputs, inputs, 10000);


            List<double> results = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.Predict(row).Output;
                results.Add(res);
            }

            double sumSquaredError = 0.0;
            for (int i = 0; i < results.Count; i++)
            {
                //if (i % 20 != 0) continue;
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Console.WriteLine($"очікуваний: {expected}, фактичний: {actual}");
                double error = expected - actual;
                sumSquaredError += error * error;
            }

            double mse = sumSquaredError / results.Count;
            double accuracy = 100.0 - (mse * 100.0);

            Console.WriteLine();
            Console.WriteLine($"Середньоквадратична помилка (MSE): {mse}");
            Console.WriteLine($"Точність: {accuracy}%");
        }
    }
}
