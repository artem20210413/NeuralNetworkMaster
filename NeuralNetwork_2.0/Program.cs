using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace NeuralNetwork_2._0
{
    internal class Program
    {
        static void Main(string[] args)
        {
            NeuronNetworkTests.FeedForwardTest();

            Console.WriteLine("...");
            Console.ReadKey(true);
        }

}
}
