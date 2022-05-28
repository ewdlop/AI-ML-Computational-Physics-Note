using AIMA.Agent.Action;
using AIMA.Agent.Percept;
using AIMA.Agent.Program;

namespace AIMA.Agent
{
    public abstract class AgentBase<T1,T2> : IAgent<T1,T2> where T1: IAction where T2: IPercept
    {
        protected readonly IAgentProgram<T1,T2> _agentProgram;
        protected AgentBase(IAgentProgram<T1, T2> agentProgram)
        {
            _agentProgram = agentProgram;
        }
        private bool Alive { get; set; }
        public virtual T1 Execute(T2 percept)
        {
            if(_agentProgram is not null)
            {
                return _agentProgram.Execute(percept);
            }
            return (T1)(NoOpAction.NO_OPERATION as IAction);
        }
        public bool IsAlive() => Alive;
        public virtual void SetAlive(bool alive) => Alive = alive;
    }
}
