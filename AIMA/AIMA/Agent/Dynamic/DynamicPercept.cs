using AIMA.Agent.Object;
using AIMA.Agent.Percept;
using System.Diagnostics;

namespace AIMA.Agent.Dynamic
{
    public class DynamicPercept : ObjectWithDynamicAttributes, IPercept
    {
        public override string DescribeType => typeof(IPercept).Name;
        public DynamicPercept(object key1, object value1)
        {
            SetAttribute(key1, value1);
        }
        public DynamicPercept(object key1, object value1, object key2, object value2)
        {
            SetAttribute(key1, value1);
            SetAttribute(key2, value2);
        }
        public DynamicPercept(object[] keys, object[] values)
        {
            Debug.Assert(keys.Length == values.Length);

            for (int i = 0; i < keys.Length; i++)
            {
                SetAttribute(keys[i], values[i]);
            }
        }
    }
}
