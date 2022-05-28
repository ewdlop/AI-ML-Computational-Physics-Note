using AIMA.Agent.Object;

namespace AIMA.Agent.Action.Dynamic
{
    public class DynamicAction : ObjectWithDynamicAttributesBase, IAction
    {
        public const string ATTRIBUTE_NAME = "name";
        public DynamicAction(string name) => SetAttribute(ATTRIBUTE_NAME, name);
        public virtual bool IsNoOperation() => false;
        public string Name => (string)GetAttribute(ATTRIBUTE_NAME);
        public override string DescribeType => GetType().Name;
    }
}
