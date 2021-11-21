using System.Collections.Generic;

namespace AIMA.Agent.Object
{
    public interface IObjectWithDynamicAttributes
    {
        IReadOnlyDictionary<object, object> ReadOnlyAttributes { get;}
        string DescribeType { get; }
        string DescribeAttributes();
        HashSet<object> GetKeySet();
        void SetAttribute(object key, object value);
        object GetAttribute(object key);
        object RemoveAttribute(object key);
        public IObjectWithDynamicAttributes Copy();
    }
}
