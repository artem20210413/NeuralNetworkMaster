using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OfficeOpenXml;

namespace NeuralNetwork_2._0
{

    public class MyDataModel
    {
        public List<double> inputs { get; set; }
        public List<double> outputs { get; set; }

        public MyDataModel(List<double> inputs, List<double> outputs) { 
            this.inputs = inputs;
            this.outputs = outputs;
        }
    }

    public class NeuralNetworkDataLoader
    {
        private string filePath;

        private char variableSeparator = ';'; // Разделитель переменных между собой
        private char ioSeparator = '|'; // Разделитель входных и выходных значений
        private int numberDefaultList = 0;

        public List<double> inputs { get; private set; }
        public List<double> outputs { get; private set; }
        public int rowCount { get; private set; }
        public int colCount { get; private set; }
        public int countColumInput { get; private set; }
        public int countColumOutput { get; private set; }
        public List<MyDataModel> data { get; private set; }

        public NeuralNetworkDataLoader(string filePath)
        {
            this.filePath = filePath;
            this.data =  ReadExcel(this.filePath);
        }

        public List<MyDataModel> ReadExcel(string filePath)
        {
            ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
            using (var package = new ExcelPackage(new FileInfo(filePath)))
            {
                var worksheet = package.Workbook.Worksheets[this.numberDefaultList]; // Получение активного листа (первого листа)
                var rowCount = worksheet.Dimension.Rows; // Получение количества строк в файле
                int colCount = worksheet.Dimension.Columns; // Получение количества столбцов в первой строке
                this.rowCount = rowCount;
                this.colCount = colCount;

                List<MyDataModel> data = new List<MyDataModel>();

                for (int row = 2; row <= rowCount; row++)
                {
                    List<double> inputs = new List<double>();
                    List<double> outputs = new List<double>();
                    //bool isInput = true;
                    int countColum = 0;

                    foreach (var cell in worksheet.Cells[row, 1, row, worksheet.Dimension.End.Column])
                    {
                        if (cell.Start.Column == 1) countColum = 0;

                        if (cell.Value == null || cell.Text == this.ioSeparator.ToString())
                        {
                            countColum++;
                            continue;
                        }

                        if (countColum == 0) inputs.Add(Convert.ToDouble(cell.Value));
                        else if (countColum == 1) outputs.Add(Convert.ToDouble(cell.Value));
                        else break;
                    }

                    if (inputs.Count == 0 || outputs.Count == 0) break;

                    if(row == 2)
                    {
                        this.countColumInput = inputs.Count;
                        this.countColumOutput = outputs.Count;
                    }

                    data.Add(new MyDataModel(inputs, outputs));
                }

                return data;
            }
        }

        public List<List<double>> getInputs()
        {

            return this.data.Select(el => el.inputs).ToList();

        }
        public List<List<double>> getOutputs()
        {
            return this.data.Select(el => el.outputs).ToList();
        }

        public List<double> ListDoubleTotoList(List<List<double>> listDouble)
        {
            List<double> flatList = new List<double>();

            foreach (var sublist in listDouble)
            {
                foreach (var value in sublist)
                {
                    flatList.Add(value);
                }
            }

            return flatList;
        }

        public double[] FlattenList(List<List<double>> listOfLists)
        {
            return listOfLists.SelectMany(list => list).ToArray();
        }

        public double[,] FlattenList2(List<List<double>> el)
        {
            int rowCount = el.Count;
            int colCount = el[0].Count;

            double[,] res = new double[rowCount, colCount];

            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < colCount; j++)
                {
                    res[i, j] = el[i][j];
                }
            }

            return res;
        }

        public dynamic ProcessListOfLists(List<List<double>> listOfLists)
        {
            if (listOfLists[0].Count == 0) 
                return null;

            if (listOfLists[0].Count == 1) 
                return FlattenList(listOfLists);

            return FlattenList2(listOfLists);
        }
    }
}
