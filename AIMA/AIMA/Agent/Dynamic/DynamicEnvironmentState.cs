using AIMA.Agent.Environment;
using AIMA.Agent.Object;

namespace AIMA.Agent.Dynamic
{
    public class DynamicEnvironmentState : ObjectWithDynamicAttributesBase, IEnvironmentState
    {
        public override string DescribeType => typeof(IEnvironmentState).Name;
    }
}
