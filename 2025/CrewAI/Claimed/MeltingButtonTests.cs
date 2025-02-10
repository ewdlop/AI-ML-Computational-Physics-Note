using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.Maui.Controls;
using System.Threading.Tasks;

[TestClass]
public class MeltingButtonTests
{
    [TestMethod]
    public async Task ButtonClick_ShouldTriggerAnimation()
    {
        var button = new MeltingButton();
        bool eventFired = false;

        button.Clicked += (sender, args) => { eventFired = true; };
        button.SendClicked();

        await Task.Delay(1000);
        Assert.IsTrue(eventFired);
    }
}
