using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_2._0
{
    internal class NeuralNetwork
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }
        private SaveWeights saveWeights;
        public long epochs { get; private set; }

        public NeuralNetwork(Topology topology)
        {
            epochs = 0;
            Topology = topology;

            Layers = new List<Layer>();
            saveWeights = new SaveWeights(topology);

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
            LoadWeight();
        }

        public Neuron Predict(List<double> inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            return Layers.Last().Neurons[0];
            //return (Topology.OutputCount == 1)
            //    ? Layers.Last().Neurons[0]
            //    : Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
        }

        public double Learn(List<double> expected, List<List<double>> inputs, int epoch)
        {
            List<double> indexRowList = this.getRandomRows(expected, 3);

            double sumError = 0.0; 
            long currentEpoch = 1;

            while (epoch == 0 || currentEpoch <= epoch)
            {

                if (Console.KeyAvailable && Console.ReadKey(intercept: true).KeyChar == 'q')  // Проверка наличия нажатия клавиши
                {
                    Console.WriteLine("Навчання перервано користувачем.");
                    break;  // Выход из цикла

                }

                var error = 0.0;

                for (int j = 0; j < expected.Count; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    var e = Backpropagation(output, input);
                    error += e;
                }

                this.preliminaryResultConsole(currentEpoch, indexRowList, inputs, expected, 1000);

                
                double currentError =   Math.Sqrt( Math.Pow(error,2) / expected.Count);
                sumError += currentError;

                if (Math.Sqrt(currentError) <= Topology.Accuracy && Topology.Accuracy != 0.0) break;

                currentEpoch++;
                epochs++;
            }

            saveWeights.saveWeights(this, epoch, epochs);

            return Math.Sqrt(Math.Pow(sumError, 2) / epoch);
        }

        private void preliminaryResultConsole(long currentEpoch, List<double> indexRowList, List<List<double>> inputs, List<double> expected, int multiplicity = 100)
        {
            if (currentEpoch % multiplicity == 1)
            {
                Console.WriteLine($"currentEpoch: {currentEpoch}");
                foreach (int indexRow in indexRowList)
                {
                    var row = GetRow(inputs, indexRow);
                    var res = this.Predict(row).Output;
                    Console.WriteLine($"очікуваний: {Math.Round(expected[indexRow], 15)}, фактичний: {Math.Round(res, 15)}");
                }
            }
        }
        private List<double> getRandomRows(List<double> expected, int count = 3)
        {
            var rand = new Random();
            var indexRowList = new List<double>();
            for (int i = 0; i < count; i++)
                indexRowList.Add(rand.Next(0, expected.Count));

            return indexRowList;
        }

        public static List<double> GetRow(List<List<double>> matrix, int row)
        {
            var list = new List<double>();

            for (int i = 0; i < matrix[row].Count; ++i)
                list.Add(matrix[row][i]);
            

            return list;
        }

        private double[,] Scaling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];

                    if (item < min)
                    {
                        min = item;
                    }

                    if (item > max)
                    {
                        max = item;
                    }
                }

                var divider = max - min;
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }

            return result;
        }

        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                // Среднее значение сигнала нейрона.
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                // Стандартное квадратичное отклонение нейрона.
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                var standardError = Math.Sqrt(error / inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standardError;
                }
            }

            return result;
        }

        private double Backpropagation(double expected, List<double> inputs)
        {
            var actual = Predict(inputs).Output;

            var difference = actual - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return Math.Pow(difference, 2);
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSingals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSingals);
                }
            }
        }

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input, this.Topology);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Normal, this.Topology);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output, this.Topology);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void LoadWeight() {
            if (!saveWeights.isSavedOne()) return;

            epochs = saveWeights.epochs;
            List<double> weights = saveWeights.weights;

            foreach (Layer layer in Layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    List<double> neuronWeights = new List<double>();
                    for (int i = 0; i < neuron.Weights.Count; i++)
                    {
                        neuronWeights.Add(weights[0]); // Получаем первый элемент списка
                        weights.RemoveAt(0);
                    }

                    neuron.LoadWeights(neuronWeights);
                }
            }

        }
    }
}
