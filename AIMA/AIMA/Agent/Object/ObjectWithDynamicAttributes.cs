using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace AIMA.Agent.Object
{
    public abstract class ObjectWithDynamicAttributes : IObjectWithDynamicAttributes
    {
        private readonly Dictionary<object, object> _attributes;
        Dictionary<object, object> IObjectWithDynamicAttributes.Attributes => _attributes;
        protected ObjectWithDynamicAttributes()
        {
            _attributes = new Dictionary<object, object>();
        }
        public virtual string DescribeType => GetType().Name;
        public virtual string DescribeAttributes()
        {
            var stringBuildr = new StringBuilder();
            stringBuildr.Append('[');
            bool first = true;
            foreach (object key in _attributes.Keys)
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
                stringBuildr.Append(_attributes[key]);
            }
            stringBuildr.Append(']');
            return stringBuildr.ToString();
        }
        public virtual HashSet<object> GetKeySet() => new(_attributes.Keys);
        public virtual void SetAttribute(object key, object value) => _attributes[key] = value;
        public virtual object GetAttribute(object key) => _attributes[key];
        public virtual object RemoveAttribute(object key) => _attributes.Remove(key);
        public virtual IObjectWithDynamicAttributes Copy()
        {
            ObjectWithDynamicAttributes copy = null;
            try
            {
                copy = (ObjectWithDynamicAttributes)GetType().GetConstructor(Type.EmptyTypes).Invoke(null);
                foreach (object value in _attributes)
                {
                    copy._attributes.Add(value, _attributes[value]);
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
            return _attributes.Equals(((ObjectWithDynamicAttributes)o)._attributes);
        }

        public override int GetHashCode()
        {
            return _attributes.GetHashCode();
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
