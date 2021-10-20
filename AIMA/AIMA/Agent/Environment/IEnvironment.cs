using AIMA.Agent.Action;
using AIMA.Agent.Percept;
using System.Collections.Generic;

namespace AIMA.Agent.Environment
{
    public interface IEnvironment<T1,T2> where T1 : IAction where T2 : IPercept
    {
        Dictionary<IAgent<T1, T2>, double> PerformanceMeasures { get; }
        IList<IAgent<T1,T2>> GetAgents();
        void AddAgent(IAgent<T1,T2> agent);
        void RemoveAgent(IAgent<T1,T2> agent);
        IList<IEnvironmentObject> GetEnvironmentObjects();
        void AddEnvironmentObject(IEnvironmentObject environmentObject);
        void RemoveEnvironmentObject(IEnvironmentObject environmentObject);
        void Step();
        void Step(int n);
        void StepUntilDone();
        bool IsDone();
        double GetPerformanceMeasure(IAgent<T1,T2> agent);
        void AddEnvironmentView(IEnvironmentView<T1,T2> environmentView);
        void RemoveEnvironmentView(IEnvironmentView<T1,T2> environmentView);
        void NotifyViews(string message);
    }
}
