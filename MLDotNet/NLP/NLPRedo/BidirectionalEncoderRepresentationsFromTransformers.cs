namespace NLPRedo;

/// <summary>
/// Using OnnxModel
/// </summary>
public class BidirectionalEncoderRepresentationsFromTransformers
{
    private readonly List<string> _vocabulary;

    private readonly Tokenizer _tokenizer;
    private readonly Predictor _predictor;

    public BidirectionalEncoderRepresentationsFromTransformers(string vocabularyFilePath, string bertModelPath)
    {
        _vocabulary = FileReader.ReadFile(vocabularyFilePath);
        _tokenizer = new Tokenizer(_vocabulary);

        var trainer = new Trainer();
        var trainedModel = trainer.BuidAndTrain(bertModelPath, false);
        _predictor = new Predictor(trainedModel);
    }

    public (List<string> tokens, float probability) Predict(string context, string question)
    {
        List<(string Token, int VocabularyIndex, long SegmentIndex)> tokens = _tokenizer.Tokenize(question, context);
        BertInput input = BuildInput(tokens);

        BertPredictions predictions = _predictor.Predict(input);

        int contextStart = tokens.FindIndex(o => o.Token == Tokens.Separation);

        (int startIndex, int endIndex, float probability) = GetBestPrediction(predictions, contextStart, 20, 30);

        List<string> predictedTokens = input.InputIds
            .Skip(startIndex)
            .Take(endIndex + 1 - startIndex)
            .Select(o => _vocabulary[(int)o])
            .ToList();

        List<string> connectedTokens = Tokenizer.Untokenize(predictedTokens);

        return (connectedTokens, probability);
    }

    private static BertInput BuildInput(List<(string Token, int Index, long SegmentIndex)> tokens)
    {
        List<long> padding = Enumerable.Repeat(0L, 256 - tokens.Count).ToList();

        long[] tokenIndexes = tokens.Select(token => (long)token.Index).Concat(padding).ToArray();
        long[] segmentIndexes = tokens.Select(token => token.SegmentIndex).Concat(padding).ToArray();
        long[] inputMask = tokens.Select(o => 1L).Concat(padding).ToArray();

        return new BertInput()
        {
            InputIds = tokenIndexes,
            SegmentIds = segmentIndexes,
            InputMask = inputMask,
            UniqueIds = new long[] { 0 }
        };
    }

    private static (int StartIndex, int EndIndex, float Probability) GetBestPrediction(BertPredictions result, int minIndex, int topN, int maxLength)
    {
        IEnumerable<(float Logit, int Index)> bestStartLogits = result.StartLogits
            .Select((logit, index) => (Logit: logit, Index: index))
            .OrderByDescending(o => o.Logit)
            .Take(topN);

        IEnumerable<(float Logit, int Index)> bestEndLogits = result.EndLogits
            .Select((logit, index) => (Logit: logit, Index: index))
            .OrderByDescending(o => o.Logit)
            .Take(topN);

        IEnumerable<(int StartLogit, int EndLogit, float Score)> bestResultsWithScore = bestStartLogits
            .SelectMany(startLogit =>
                bestEndLogits
                .Select(endLogit =>
                    (
                        StartLogit: startLogit.Index,
                        EndLogit: endLogit.Index,
                        Score: startLogit.Logit + endLogit.Logit
                    )
                 )
            )
            .Where(entry => !(entry.EndLogit < entry.StartLogit
                              || entry.EndLogit - entry.StartLogit > maxLength
                              || entry.StartLogit == 0
                              && entry.EndLogit == 0
                              || entry.StartLogit < minIndex))
            .Take(topN);

        ((int StartLogit, int EndLogit, float Score) item, float probability) = bestResultsWithScore
            .Softmax(o => o.Score)
            .OrderByDescending(o => o.Probability)
            .FirstOrDefault();

        return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
    }
}