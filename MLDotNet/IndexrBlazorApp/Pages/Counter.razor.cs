using Lucene.Net.Analysis;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;
using System.Reflection;

namespace IndexrBlazorApp.Pages;

public partial class Counter
{
    public const LuceneVersion AppLuceneVersion = LuceneVersion.LUCENE_48;
    private const string Name = "name";
    private const string FavoritePhrase = "favoritePhrase";
    private StandardAnalyzer? _analyzer;
    protected override void OnInitialized()
    {
        _analyzer = new(AppLuceneVersion);
        IndexWriterConfig indexConfig = new(AppLuceneVersion, _analyzer);
        string indexPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? string.Empty, "index");
        using FSDirectory dir = FSDirectory.Open(indexPath);
        using IndexWriter writer = new IndexWriter(dir, indexConfig);
        // Search with a phrase
        MultiPhraseQuery phraseQuery = new MultiPhraseQuery
        {
            //new Term("favoritePhrase", "quick"),
            //new Term("favoritePhrase", "brown"),
            //new Term("favoritePhrase", "fox"),
            //new Term("favoritePhrase", "jumps")
            new Term(Counter.Name,"Kermit the Frog")
        };
        (string Name, string FavoritePhrase) =
        (
            "Kermit the Frog", 
            "The quick brown fox jumps over the lazy dog"
        );
        var doc = new Document
        {
            // StringField indexes but doesn't tokenize
            new StringField(Counter.Name, 
                Name,
                Field.Store.YES),
            new TextField(Counter.FavoritePhrase, 
                FavoritePhrase,
                Field.Store.YES),
        };
        writer.AddDocument(doc);
        writer.Flush(triggerMerge: false, applyAllDeletes: false);
        using DirectoryReader reader = writer.GetReader(applyAllDeletes: true);
        IndexSearcher searcher = new IndexSearcher(reader);
        ScoreDoc[] hits = searcher.Search(phraseQuery, 20 /* top 20 */).ScoreDocs;

        // Display the output in a table
        Console.WriteLine($"{"Score",10}" +
            $" {"Name",-15}" +
            $" {"Favorite Phrase",-40}");
        foreach (var hit in hits)
        {
            var foundDoc = searcher.Doc(hit.Doc);
            Console.WriteLine($"{hit.Score:f8}"
                + $" {foundDoc.Get(Counter.Name),-15}"
                + $" {foundDoc.Get(Counter.FavoritePhrase),-40}");
        }
        base.OnInitialized();
    }

    private int currentCount = 0;
    private void IncrementCount()
    {
        currentCount++;
    }
}