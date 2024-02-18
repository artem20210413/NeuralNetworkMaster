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

            //List<double> outputs = new List<double> { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            List<List<double>> outputs = new List<List<double>>
            {
                new List<double>  { 0,1},
                new List<double>  { 0,1},
                new List<double>  { 1,1},
                new List<double>  { 0,1},
                new List<double>  { 0,1},
                new List<double>  { 0,1},
                new List<double>  { 1,1},
                new List<double>  { 0,1},
                new List<double>  { 1,1},
                new List<double>  { 1,1},
                new List<double>  { 1,1},
                new List<double>  { 1,1},
                new List<double>  { 1,1},
                new List<double>  { 0,1},
                new List<double>  { 1,1},
                new List<double>  { 1,1}
            };

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
                2,
                0.1,
                5,5
                );
            topology.EnumActivationFunction = EnumActivationFunction.Sigmoid;

            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            double difference = neuralNetwork.Learn(outputs, inputs, 1000);


            List<List<double>> results = new List<List<double>>();
            for (int i = 0; i < outputs.Count; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.Predict(row).Neurons;

                List<double> rowOutput = new List<double>();
                foreach ( var r in res)
                {
                    rowOutput.Add(r.Output);
                }

                results.Add(rowOutput);
            }

            double sumSquaredError = 0.0;
            for (int i = 0; i < outputs.Count; i++)
            {
                //if (i % 20 != 0) continue;

                for (int j = 0; j < outputs[i].Count; j++)
                {
                    var expected = Math.Round(outputs[i][j], 2);
                    var actual = Math.Round(results[i][j], 2);
                    Console.Write($"очікуваний[{j}]: {expected}\t");
                    double error = expected - actual;
                    sumSquaredError += error * error;
                }

                Console.Write("|\t");

                for (int j = 0; j < outputs[i].Count; j++)
                {
                    var actual = Math.Round(results[i][j], 2);
                    Console.Write($"фактичний[{j}]: {actual}\t");
                }

                Console.WriteLine();

            }

            double mse = sumSquaredError / results.Count;
            double accuracy = 100.0 - (mse * 100.0);

            Console.WriteLine();
            Console.WriteLine($"Середньоквадратична помилка (MSE): {mse}");
            Console.WriteLine($"Точність: {accuracy}%");
        }
    }
}
