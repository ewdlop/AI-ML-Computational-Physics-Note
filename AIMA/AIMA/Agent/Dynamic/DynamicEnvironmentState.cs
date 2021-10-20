using AIMA.Agent.Environment;
using AIMA.Agent.Object;

namespace AIMA.Agent.Dynamic
{
    public class DynamicEnvironmentState : ObjectWithDynamicAttributes, IEnvironmentState
    {
        public override string DescribeType => typeof(IEnvironmentState).Name;
    }
}
