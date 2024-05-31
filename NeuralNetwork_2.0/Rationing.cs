using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_2._0
{
    internal class DataRationing
    {
       private List<double> limitColumn = new List<double> { 11, 260, 275, 2600, 50, 1500 };
        public List<List<double>> rationingInputs { get; private set; }
        public List<double> rationingOutputs { get; private set; }

        NeuralNetworkDataLoader networkDataLoader;

        public DataRationing(NeuralNetworkDataLoader networkDataLoader)
        {
            this.networkDataLoader = networkDataLoader;
            rationingInputs = RationingInputs();
            rationingOutputs = RationingOutputs();
        }

        private double RationingOutput(double output, int row)
        {
            return output / DeRationingInput(rationingInputs[row][4], 4);
        }

        public double DeRationingOutput(double output, int row)
        {
            return output * DeRationingInput(rationingInputs[row][4], 4);
        }

        private double RationingInput(double input, int Column)
        {
            return input / limitColumn[Column];
        } 

        private double DeRationingInput(double input, int Column)
        {
            return input * limitColumn[Column];
        }

        private List<List<double>> RationingInputs()
        {
            List<List<double>> resInputs = new List<List<double>>();
            foreach (List<double> input in networkDataLoader.getInputs())
            {

                List<double> resInput = new List<double>();
                int i = 0;
                foreach (double value in input)
                {
                    resInput.Add(RationingInput(value, i));
                    i++;
                }
                resInputs.Add(resInput);
            }

            return resInputs;
        }
        
        private List<double> RationingOutputs()
        {
            List<double> resOutputs = new List<double>();
            int row = 0;
            foreach (double output in networkDataLoader.getOutputsList())
            {
                resOutputs.Add(RationingOutput(output, row));
                row++;
            }

            return resOutputs;
        }




        //public List<List<double>> input { get; private set; }
        //public List<List<double>> output { get; private set; }
        //public double sheetThickness { get; private set; }


        //public Rationing(double sheetThickness)
        //{
        //    this.sheetThickness = sheetThickness;
        //}



        //public double rationingOutput(double outputs)
        //{
        //    return outputs / sheetThickness;
        //}

        //public double derationingOutput(double output)
        //{
        //    return output * sheetThickness;
        //}
    }
}
