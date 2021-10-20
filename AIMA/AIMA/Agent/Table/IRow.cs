using System.Collections.Generic;

namespace AIMA.Agent.Table
{
    public interface IRow<ColumnHeaderType, ValueType> where ValueType:struct
    {
        Dictionary<ColumnHeaderType, ValueType> Columns { get; init; }

        public Dictionary<ColumnHeaderType, ValueType> Cells()
        {
            return Columns;
        }

    }
}
