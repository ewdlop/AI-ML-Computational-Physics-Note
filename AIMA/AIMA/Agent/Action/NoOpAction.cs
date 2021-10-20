using AIMA.Agent.Action.Dynamic;

namespace AIMA.Agent.Action
{
    public sealed class NoOpAction : DynamicAction
    {
        public static readonly NoOpAction NO_OPERATION = new();
        public override bool IsNoOperation() => true;
        private NoOpAction() : base("NoOp") {}
    }
}
