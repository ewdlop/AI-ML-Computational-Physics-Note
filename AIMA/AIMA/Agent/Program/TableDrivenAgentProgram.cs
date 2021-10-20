using AIMA.Agent.Action;
using AIMA.Agent.Percept;
using System.Collections.Generic;

namespace AIMA.Agent.Program
{
    public class TableDrivenAgentProgram<T1,T2> : IAgentProgram<T1,T2> where T1: IAction where T2:IPercept
    {
        private IList<T2> Percepts { get;init; } = new List<T2>();
        //private Table<List<Percept>, System.String, Action> table;
        private const string ACTION = "action";

        public T1 Execute(T2 percept)
        {
            // append percept to end of percepts
            Percepts.Add(percept);

            // action <- LOOKUP(percepts, table)
            // return action
            //return LookupCurrentAction();

            return default;
        }
    }
}
