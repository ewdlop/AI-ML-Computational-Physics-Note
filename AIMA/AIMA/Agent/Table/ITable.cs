using System.Collections.Generic;

namespace AIMA.Agent.Table
{
    public interface ITable<RowHeaderType, ColumnHeaderType, ValueType> where ValueType : struct
    {
        List<RowHeaderType> RowHeaders { get; init; }
        List<ColumnHeaderType> ColumnHeaders { get; init; }
        Dictionary<RowHeaderType, Dictionary<ColumnHeaderType, ValueType>> Rows { get; init; }
        void Set(RowHeaderType r, ColumnHeaderType c, ValueType v);
        ValueType Get(RowHeaderType r, ColumnHeaderType c);
    }
}
