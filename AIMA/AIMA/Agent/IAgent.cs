using AIMA.Agent.Action;
using AIMA.Agent.Environment;
using AIMA.Agent.Percept;

namespace AIMA.Agent
{
    public interface IAgent<T1,T2> : IEnvironmentObject
        where T1:IAction
        where T2: IPercept
    {
        T1 Execute(T2 percept);
        bool IsAlive();
        void SetAlive(bool alive);
    }
}
