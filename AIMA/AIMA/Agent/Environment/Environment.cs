using AIMA.Agent.Action;
using AIMA.Agent.Percept;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIMA.Agent.Environment
{
    public abstract class Environment<T1, T2> : IEnvironmentViewNotifier, IEnvironment<T1, T2>
        where T1 : IAction
        where T2 : IPercept
    {
        //protected LinkedHashSet<EnvironmentObject> envObjects = new LinkedHashSet<EnvironmentObject>();

        //protected LinkedHashSet<Agent> agents = new LinkedHashSet<Agent>();

        //protected LinkedHashSet<EnvironmentView> views = new LinkedHashSet<EnvironmentView>();

        protected readonly Dictionary<IAgent<T1, T2>, double> _performanceMeasures;
        Dictionary<IAgent<T1, T2>, double> IEnvironment<T1, T2>.PerformanceMeasures => _performanceMeasures;
        protected Environment()
        {
            _performanceMeasures = new();
        }
        public abstract IEnvironmentState GetCurrentState();
        public abstract IEnvironmentState ExecuteAction(IAgent<T1, T2> agent, T1 action);
        public abstract T2 GetPerceptSeenBy(IAgent<T1, T2> agent);
        public virtual IList<IAgent<T1, T2>> GetAgents()
        {
            // Return as a List but also ensures the caller cannot modify
            //return new List<IAgent<T1, T2>>(agents);
            return null;
        }

        public virtual void AddAgent(IAgent<T1, T2> agent) => AddEnvironmentObject(agent);

        public virtual void RemoveAgent(IAgent<T1, T2> agent) => RemoveEnvironmentObject(agent);
        public virtual IList<IEnvironmentObject> GetEnvironmentObjects()
        {
            //// Return as a List but also ensures the caller cannot modify
            //return new List<IEnvironmentObject>(envObjects);
            return null;
        }

        public virtual void AddEnvironmentObject(IEnvironmentObject environmentObject)
        {
            //envObjects.Add(eo);
            if (environmentObject is IAgent<T1, T2>)
            {
                //if (!agents.Contains(environmentObject as IAgent<T1, T2>))
                //{
                //    agents.Add(a);
                //    this.updateEnvironmentViewsAgentAdded(a);
                //}
            }
        }

        public virtual void RemoveEnvironmentObject(IEnvironmentObject environmentObject)
        {
            //envObjects.Remove(eo);
            //agents.Remove(eo);
        }

        public virtual void Step()
        {
            //foreach (IAgent<T1, T2> agent in agents)
            //{
            //    if (agent.isAlive())
            //    {
            //        Action anAction = agent.execute(getPerceptSeenBy(agent));
            //        EnvironmentState es = executeAction(agent, anAction);
            //        updateEnvironmentViewsAgentActed(agent, anAction, es);
            //    }
            //}
            CreateExogenousChange();
        }

        public virtual void Step(int n)
        {
            for (int i = 0; i < n; i++)
            {
                Step();
            }
        }

        public virtual void StepUntilDone()
        {
            while (!IsDone())
            {
                Step();
            }
        }

        public virtual bool IsDone()
        {
            //foreach (Agent agent in agents)
            //{
            //    if (agent.isAlive())
            //    {
            //        return false;
            //    }
            //}
            return true;
        }

        public virtual double GetPerformanceMeasure(IAgent<T1, T2> agent)
        {
            double? performanceMeasures = _performanceMeasures[agent];
            if (!performanceMeasures.HasValue)
            {
                performanceMeasures = 0.0;
                _performanceMeasures[agent] = performanceMeasures.Value;
            }
            return performanceMeasures.Value;
        }

        public virtual void AddEnvironmentView(IEnvironmentView<T1, T2> environmentView)
        {
            //views.Add(environmentView);
        }

        public virtual void RemoveEnvironmentView(IEnvironmentView<T1, T2> environmentView)
        {
            //views.Remove(ev);
        }
        public virtual void NotifyViews(string msg)
        {
            throw new NotImplementedException();
        }
        void CreateExogenousChange()
        {

        }
        void UpdatePerformanceMeasure(IAgent<T1, T2> forAgent, double addTo)
        {

        }
        void UdateEnvironmentViewsAgentAdded(IAgent<T1, T2> agent)
        {

        }
        protected virtual void UpdateEnvironmentViewsAgentActed(IAgent<T1, T2> agent, T1 action,
                IEnvironmentState state)
        {

        }
    }
}
