using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIMA.Agent.Table
{
    public abstract record Table<RowHeaderType, ColumnHeaderType, ValueType>(List<RowHeaderType> RowHeaders,
        List<ColumnHeaderType> ColumnHeaders, Dictionary<RowHeaderType, Dictionary<ColumnHeaderType, ValueType>> Rows)
        : ITable<RowHeaderType, ColumnHeaderType, ValueType> where ValueType : struct
    {

        public virtual ValueType Get(RowHeaderType r, ColumnHeaderType c)
        {
            return default;
        }
        public virtual void Set(RowHeaderType r, ColumnHeaderType c, ValueType v)
        { 
        }
    }
}
