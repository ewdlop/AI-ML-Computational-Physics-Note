﻿using System.Collections.Generic;

namespace AIMA.Agent.Object
{
    public interface IObjectWithDynamicAttributes
    {
        Dictionary<object, object> Attributes { get;}
        string DescribeType { get; }
        string DescribeAttributes();
        HashSet<object> GetKeySet();
        void SetAttribute(object key, object value);
        object GetAttribute(object key);
        object RemoveAttribute(object key);
        public IObjectWithDynamicAttributes Copy();
    }
}