using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text;

namespace AIMA.Agent.Object
{
    public abstract class ObjectWithDynamicAttributes : IObjectWithDynamicAttributes
    {
        private Dictionary<object, object> Attributes { get; init; } = new();
        public IReadOnlyDictionary<object, object> ReadOnlyAttributes => Attributes;
        public virtual string DescribeType => GetType().Name;
        public virtual string DescribeAttributes()
        {
            var stringBuildr = new StringBuilder('[');
            bool first = true;
            foreach (object key in Attributes.Keys)
            {
                if (first)
                {
                    first = false;
                }
                else
                {
                    stringBuildr.Append(", ");
                }
                stringBuildr.Append(key);
                stringBuildr.Append("==");
                stringBuildr.Append(Attributes[key]);
            }
            stringBuildr.Append(']');
            return stringBuildr.ToString();
        }
        public virtual HashSet<object> GetKeySet() => new(Attributes.Keys);
        public virtual void SetAttribute(object key, object value) => Attributes[key] = value;
        public virtual object GetAttribute(object key) => Attributes[key];
        public virtual object RemoveAttribute(object key) => Attributes.Remove(key);
        public virtual IObjectWithDynamicAttributes Copy()
        {
            ObjectWithDynamicAttributes copy = null;
            try
            {
                copy = (ObjectWithDynamicAttributes)GetType().GetConstructor(Type.EmptyTypes).Invoke(null);
                foreach (object value in Attributes)
                {
                    copy.Attributes.Add(value, Attributes[value]);
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
            }
            return copy;
        }

        public override bool Equals(object o)
        {
            if (o == null || GetType() != o.GetType())
            {
                return base.Equals(o);
            }
            return Attributes.Equals(((ObjectWithDynamicAttributes)o).Attributes);
        }

        public override int GetHashCode()
        {
            return Attributes.GetHashCode();
        }

        public override string ToString()
        {
            var stringBuilder = new StringBuilder();
            stringBuilder.Append(DescribeType);
            stringBuilder.Append(DescribeAttributes());
            return stringBuilder.ToString();
        }
    }
}
