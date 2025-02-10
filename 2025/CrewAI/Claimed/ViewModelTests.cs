using System.ComponentModel;
using Xunit;

public class ViewModelTests
{
    [Fact]
    public void PropertyChangedEvent_Fires_When_PropertyChanges()
    {
        var viewModel = new SampleViewModel();
        bool eventFired = false;

        viewModel.PropertyChanged += (sender, args) =>
        {
            if (args.PropertyName == "Name")
                eventFired = true;
        };

        viewModel.Name = "New Value";
        Assert.True(eventFired);
    }
}

public partial class SampleViewModel
{
    private string _name;
    public string Name
    {
        get => _name;
        set
        {
            _name = value;
            OnPropertyChanged(nameof(Name));
        }
    }
}
