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

            NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_2_1/dataSetRationing_training.xlsx");
            //NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_9_1/dataSetRationing_training.xlsx");
            var inputs = neuralnetworkdataloader.getInputs();
            var outputsList = neuralnetworkdataloader.getOutputs();
            var outputs = neuralnetworkdataloader.ListDoubleTotoList(outputsList);

            int[] hiddenLayers = { 30, 10 };
            string hiddenLayersString = string.Join(", ", hiddenLayers);
            Console.WriteLine($"Введіть кількість нейронів для кожного прихованого шару через кому (стандарт:, {hiddenLayersString}):");
            string input = Console.ReadLine();
            if (!string.IsNullOrEmpty(input))
                hiddenLayers = input.Split(',').Select(int.Parse).ToArray();



            Topology topology = new Topology(
                neuralnetworkdataloader.countColumInput,
                neuralnetworkdataloader.countColumOutput,
                0.1, // learningRate
               hiddenLayers); // hidden layer

            topology.EnumActivationFunction = EnumActivationFunction.Sigmoid;
            topology.ResetSave = false; // false - до навчаємо


            int epoch = 10000;
            Console.WriteLine($"Введіть кількість епох (стандарт: {epoch}):");
            string inp = Console.ReadLine();
            if (!string.IsNullOrEmpty(inp))
                epoch = Convert.ToInt32(inp);

            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            double difference = neuralNetwork.Learn(outputs, inputs, epoch);

            //topology.Accuracy = 0.001;
            List<double> results = new List<double>();


            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("Тестові дані:");

            NeuralNetworkDataLoader neuralnetworkdataloaderTest = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_2_1/dataSetRationing_test.xlsx");
            //NeuralNetworkDataLoader neuralnetworkdataloaderTest = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_9_1/dataSetRationing_test.xlsx");

            var inputsTest = neuralnetworkdataloaderTest.getInputs();
            var outputsTest = neuralnetworkdataloaderTest.ListDoubleTotoList(neuralnetworkdataloaderTest.getOutputs());
            for (int i = 0; i < outputsTest.Count; i++)
            {
                var row = NeuralNetwork.GetRow(inputsTest, i);
                var res = neuralNetwork.Predict(row).Output;
                results.Add(res);
            }

            //double sumSquaredError = 0.0;
            double sumError = 0.0;
            double sumActual = 0.0;
            for (int i = 0; i < results.Count; i++)
            {
                if (i % 1 != 0) continue;
                double expected = Math.Round(outputsTest[i], 10);
                double actual = Math.Round(results[i], 10);
                sumError += Math.Abs(actual - expected);
                sumActual += actual;

                //Rationing rationing = new Rationing(15);
                //actual = rationing.derationingOutput(actual);
                //expected = rationing.derationingOutput(expected);


                Console.WriteLine($"очікуваний: {expected}, фактичний: {actual}");

                //CalculationW c = new CalculationW(actual);
                //c.toConsole();
            }

            double mse = Math.Pow(sumError / results.Count, 2);
            double accuracy = 100.0 - (mse * 100.0);
            double accuracyExpected = Math.Abs(sumError * 100 / sumActual - 100);

            Console.WriteLine();
            Console.WriteLine($"Кількість пройдених епох: {neuralNetwork.epochs}");
            Console.WriteLine($"Середньоквадратична помилка (MSE): {mse}");
            Console.WriteLine($"Точність (середньоквадратична помилка): {accuracy}%");
            Console.WriteLine($"Точність (фактична): {accuracyExpected}%");
        }

        //static private double leveling(double expected, double actual)
        //{
        //    Random random = new Random();
        //    double errorPercentage = random.Next(1, 10) / 100.0; // Рандомне число від 0.01 до 0.05
        //    double error1 = expected * errorPercentage;

        //    // Генеруємо фактичне значення з урахуванням похибки

        //    return expected + (random.NextDouble() * 2 - 1) * error1; // Генеруємо випадкове значення з випадковим знаком в межах похибки
        //}
    }


    class prod
    {
        static public void build()
        {
            //NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_2_1/dataSetRationing_training.xlsx");
            NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSetFull/dataSet.xlsx");
            DataRationing dataRationing = new DataRationing(neuralnetworkdataloader);
            Topology topology = createTopology(neuralnetworkdataloader);
            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            training(neuralNetwork, dataRationing);
            test(neuralNetwork, dataRationing);
        }
        static public void build_row()
        {
            //NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_2_1/dataSetRationing_training.xlsx");
            NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSetFull/dataSet.xlsx");
            DataRationing dataRationing = new DataRationing(neuralnetworkdataloader);
            Topology topology = createTopology(neuralnetworkdataloader);
            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            row(neuralNetwork, dataRationing);
        }

        static public Topology createTopology(NeuralNetworkDataLoader neuralnetworkdataloader)
        {

            Console.WriteLine("Створення топології...");

            int[] hiddenLayers = { 30, 10 };
            string hiddenLayersString = string.Join(", ", hiddenLayers);
            Console.WriteLine($"Введіть кількість нейронів для кожного прихованого шару через кому (стандарт:, {hiddenLayersString}):");
            string input = Console.ReadLine();
            if (!string.IsNullOrEmpty(input))
                hiddenLayers = input.Split(',').Select(int.Parse).ToArray();

            double learningRate = 0.1;
            Console.WriteLine($"Введіть швідкість навчання (стандарт:, {learningRate}):");
            input = Console.ReadLine();
            if (!string.IsNullOrEmpty(input))
                learningRate = Convert.ToDouble(input);

            Topology topology = new Topology(
            neuralnetworkdataloader.countColumInput,
            neuralnetworkdataloader.countColumOutput,
            learningRate, // learningRate
            hiddenLayers); // hidden layer

            topology.EnumActivationFunction = EnumActivationFunction.Sigmoid;
            topology.ResetSave = false; // false - до навчаємо

            topology.consoleInfo();

            return topology;
        }

        static public NeuralNetwork training(NeuralNetwork neuralNetwork, DataRationing dataRationing)
        {
            Console.WriteLine();

            //NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_2_1/dataSetRationing_training.xlsx");
            ////NeuralNetworkDataLoader neuralnetworkdataloader = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_9_1/dataSetRationing_training.xlsx");
            //var inputs = neuralnetworkdataloader.getInputs();
            ////var outputsList = neuralnetworkdataloader.getOutputs();
            //var outputs = neuralnetworkdataloader.getOutputsList();


            int epoch = 10000;
            Console.WriteLine($"Введіть кількість епох для навчання (стандарт: {epoch}):");
            string inp = Console.ReadLine();
            if (!string.IsNullOrEmpty(inp))
                epoch = Convert.ToInt32(inp);



            Console.WriteLine("Щоб зупинити навчання натисніть Q.");
            Console.WriteLine("Навчання...");
            double difference = neuralNetwork.Learn(dataRationing.rationingOutputs, dataRationing.rationingInputs, epoch);

            return neuralNetwork;
        }

        static public void predict(NeuralNetwork neuralNetwork)
        {

        }

        static public void test(NeuralNetwork neuralNetwork, DataRationing dataRationing)
        {
            Console.WriteLine();
            Console.WriteLine("Тестові дані:");
            
            List<double> results = new List<double>();
            //NeuralNetworkDataLoader neuralnetworkdataloaderTest = new NeuralNetworkDataLoader("NeuralNetworkData/DataSet_2_1/dataSetRationing_test.xlsx");

            //var inputsTest = neuralnetworkdataloaderTest.getInputs();
            //var outputsTest = neuralnetworkdataloaderTest.ListDoubleTotoList(neuralnetworkdataloaderTest.getOutputs());
            var inputsTest = dataRationing.rationingInputs;
            var outputsTest = dataRationing.rationingOutputs;
            //for (int i = 0; i < outputsTest.Count; i++)
            //{
            //    var row = NeuralNetwork.GetRow(inputsTest, i);
            //    var res = neuralNetwork.Predict(row).Output;
            //    results.Add(res);
            //}
            Random random = new Random();
            for (int i = 0; i < 15; i++)
            {
                // Генерируем случайный индекс
                int randomIndex = random.Next(outputsTest.Count);

                // Получаем строку и результат для данного индекса
                var row = NeuralNetwork.GetRow(inputsTest, randomIndex);
                var res = neuralNetwork.Predict(row).Output;

                // Добавляем результат в список
                results.Add(res);
            }
            //double sumSquaredError = 0.0;
            double sumError = 0.0;
            double sumActual = 0.0;
            for (int i = 0; i < results.Count; i++)
            {
                if (i % 1 != 0) continue;
                double expected = Math.Round(outputsTest[i], 10);
                double actual = Math.Round(results[i], 10);
                sumError += Math.Abs(actual - expected);
                sumActual += actual;

                Console.WriteLine($"очікуваний: {expected}, фактичний: {actual}");

            }

            double mse = Math.Pow(sumError / results.Count, 2);
            double accuracy = 100.0 - (mse * 100.0);
            double accuracyExpected = Math.Abs(sumError * 100 / sumActual - 100);

            Console.WriteLine();
            Console.WriteLine($"Кількість пройдених епох: {neuralNetwork.epochs}");
            Console.WriteLine($"Середньоквадратична помилка (MSE): {mse}");
            Console.WriteLine($"Точність (середньоквадратична помилка): {accuracy}%");
            Console.WriteLine($"Точність (фактична): {accuracyExpected}%");


        }

        static public void row(NeuralNetwork neuralNetwork, DataRationing dataRationing)
        {
            Console.WriteLine();
            Console.WriteLine("Введіть рядок з якого считувати дані:");
            int rowNumber = Convert.ToInt32(Console.ReadLine());

            var inputsTest = dataRationing.rationingInputs;
            var outputsTest = dataRationing.rationingOutputs;

            var row = NeuralNetwork.GetRow(inputsTest, rowNumber);
            double res = neuralNetwork.Predict(row).Output;
            res = dataRationing.DeRationingOutput(res, rowNumber);
            Console.WriteLine();
            Console.WriteLine($"Результат:");

            CalculationW c = new CalculationW(res);
            c.toConsole();
        }
    }
}
