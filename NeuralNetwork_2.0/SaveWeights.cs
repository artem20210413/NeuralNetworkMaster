using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;

namespace NeuralNetwork_2._0
{
    internal class SaveWeights
    {

        public string filePath { get; private set; }
        public char separator = '-';
        public Topology topology {  get; private set; }
        public long epochs { get; private set; }
        public List<double> weights { get; private set; }

        public SaveWeights(Topology topology) {

            this.topology = topology;

            generationPath();
            deleteFile();

            if(isSavedOne())
                weights = GetWeightsFromFile();
        }

        public bool isSavedOne()
        {
            bool isFile = File.Exists(filePath);

            return isFile && !topology.ResetSave;
        }

        public void deleteFile()
        {
            if (File.Exists(filePath) && topology.ResetSave) { 
                File.Delete(filePath); 
            }
        }

        private void generationPath()
        {
            string key  = Convert.ToString(topology.EnumActivationFunction) + separator;
            key  += Convert.ToString(topology.InputCount) + separator;
            foreach (var item in topology.HiddenLayers)
            {

                key += Convert.ToString(item) + separator;
            }
            key += Convert.ToString(topology.OutputCount);
            filePath = $"save_weights/{key}.txt";
        }


        public void saveWeights(NeuralNetwork neuralNetwork, int epoch, long epochs)
        {
            if (epoch == 0) return;
            try
            {
                // Создаем новый файл для записи весов
                using (StreamWriter writer = new StreamWriter(filePath))
                {
                    writer.WriteLine(epochs);

                    foreach (Layer layer in neuralNetwork.Layers)
                    {
                        foreach (Neuron neuron in layer.Neurons)
                        {
                            foreach (double weight in neuron.Weights)
                            {
                                // Записываем каждый вес на отдельной строке
                                writer.WriteLine(weight);
                            }
                        }
                    }
                    writer.Close();
                }

                Console.WriteLine("Ваги успішно збережені у файл.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Помилка при збереженні ваг: " + ex.Message);
            }
        }

        public List<double> GetWeightsFromFile()
        {
            List<double> weights = new List<double>();
            int epochs = 0;
            try
            {
                
                using (StreamReader reader = new StreamReader(filePath)) // Открываем файл для чтения
                {
                    string line = reader.ReadLine(); // Читаем первую строку отдельно

                    if (int.TryParse(line, out epochs)) // Обрабатываем первую строку как количество эпох
                        this.epochs = epochs;
                    else
                        Console.WriteLine("Помилка у значенні кількості епох: " + line);


                    while ((line = reader.ReadLine()) != null)  // Читаем файл построчно
                    {   
                        if (double.TryParse(line, out double weight))  // Пробуем конвертировать строку в число и добавляем в список
                        {
                            weights.Add(weight);
                        }
                        else
                            Console.WriteLine("Некоректне значення ваги у файлі: " + line);
                        
                    }
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine("Помилка при читанні ваг з файлу: " + ex.Message);
            }

            return weights;
        }


    }
}
