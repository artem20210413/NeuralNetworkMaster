using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_2._0
{
    internal class CalculationW
    {

        public List<double> ws { get; private set; }
        public int rollersCount { get; private set; }
        public int rollerPitch { get; private set; }

        public CalculationW(double w3, int rollersCount = 11, int rollerPitch = 275)
        {
            this.rollersCount = rollersCount;
            this.rollerPitch = rollerPitch;
            ws = new List<double>();

            for (int i = 1; i <= rollersCount; i++)
            {
               
                if (i == 3)
                    ws.Add(w3);
                else if (i % 2 == 0 || i == 1)
                    ws.Add(0.0);
                else
                {
                    int q = (rollersCount - i) / 2;
                    ws.Add(w3 * q * rollerPitch / (4 * rollerPitch));
                }
            }
        }

        public void toConsole()
        {
            int i = 1;
            foreach (var w in ws)
            {
                Console.Write($"W[{i}]={w}. ");
                i++;
            }
            Console.WriteLine();
        }

    }
}
