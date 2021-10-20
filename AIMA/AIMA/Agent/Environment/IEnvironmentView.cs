using AIMA.Agent.Action;
using AIMA.Agent.Percept;

namespace AIMA.Agent.Environment
{
    public interface IEnvironmentView<T1,T2>
        where T1:IAction
        where T2: IPercept
    {
        void OnNotified(string message);
        void OnAgentAdded(IAgent<T1,T2> agent, IEnvironmentState resultingState);
        void OnAgentActed(IAgent<T1,T2> agent, T1 action, IEnvironmentState resultingState);
    }
}