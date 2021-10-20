using AIMA.Agent.Action;
using AIMA.Agent.Percept;

namespace AIMA.Agent.Program
{
    public interface IAgentProgram<T1,T2>
        where T1:IAction
        where T2:IPercept
    {
        T1 Execute(T2 percept);
    }
}
