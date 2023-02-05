using Examine;
using Microsoft.AspNetCore.Components;

namespace IndexrBlazorApp.Pages;

public partial class Index
{
    [Inject] public required IExamineManager ExamineManager { get; init; }
    private const string IndexName = "MyIndex";
    private const string Category = "MyCategory";
    private IIndex? SearchIndex;
    protected override void OnInitialized()
    {
        //ExamineManager.Cr
        if (ExamineManager.TryGetIndex(IndexName, out SearchIndex))
        {
            // Add a "ValueSet" (document) to the index 
            // which can contain any data you want.
            SearchIndex.IndexItem(new ValueSet(
                Guid.NewGuid().ToString(),  //Give the doc an ID of your choice
                Category,               //Each doc has a "Category"
                new Dictionary<string, object>()
                {
                    {"Name", "Frank" },
                    {"Address", "Beverly Hills, 90210" }
                }));
        }
        base.OnInitialized(); 
    }

    private void OnSearch()
    {
        if (SearchIndex is null) return;
        ISearcher searcher = SearchIndex.Searcher; // Get a searcher
        ISearchResults results = searcher.CreateQuery()  // Create a query
            .Field("Address", "Hills")        // Look for any "Hills" addresses
            .Execute();
        
    }
}