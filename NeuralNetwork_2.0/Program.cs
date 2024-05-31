using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace NeuralNetwork_2._0
//namespace NeuralNetwork_2._0_
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.Unicode;

            //prod.build();

            prod.build_row();
            //NeuronNetworkTests.FeedForwardTest();

            Console.WriteLine("...");
            Console.ReadKey(true);
        }

    }
}
