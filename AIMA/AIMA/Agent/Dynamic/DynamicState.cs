using AIMA.Agent.Object;
using AIMA.Agent.State;

namespace AIMA.Agent.Dynamic
{
    public class DynamicState : ObjectWithDynamicAttributes, IState
    {
        public override string DescribeType => typeof(IState).Name;
    }
}
