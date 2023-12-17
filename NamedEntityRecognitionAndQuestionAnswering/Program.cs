using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.Roberta;
using Microsoft.ML.Transforms;
using System.Runtime.InteropServices;

// training data
int maxEpochs = 10;
int batchSize = 32;
int topK = 1;

// QA training data
string contextColumnName = "Context";
string questionColumnName = "Question";
string trainingAnswerColumnName = "Answer";
string answerIndexColumnName = "AnswerIndex";
string predictedAnswerColumnName = "PredictedAnswer";
string scoreColumnName = "Score";

// NER training data
string labelColumnName = "Label";
string outputColumnName = "Output";
string sentence1ColumnName = "Sentence1";
string inputColumn = "Input";
string outputColumn = "Output";

// QA trainer
MLContext mlContext = new MLContext();
EstimatorChain<ITransformer> chain = new EstimatorChain<ITransformer>();
EstimatorChain<QATransformer> estimatorQA = chain.Append(mlContext.MulticlassClassification.Trainers.QuestionAnswer(
    contextColumnName,
    questionColumnName,
    trainingAnswerColumnName,
    answerIndexColumnName,
    predictedAnswerColumnName,
    scoreColumnName,
    topK,
    batchSize,
    maxEpochs,
    Microsoft.ML.TorchSharp.NasBert.BertArchitecture.Roberta,
    null));

// NER trainer
EstimatorChain<KeyToValueMappingTransformer> estimatorNER = chain.Append(mlContext.Transforms.Conversion.MapValueToKey("Label", inputColumn))
    .Append(mlContext.MulticlassClassification.Trainers.NameEntityRecognition(
        labelColumnName,
        outputColumnName,
        sentence1ColumnName,
        batchSize,
        maxEpochs,
        Microsoft.ML.TorchSharp.NasBert.BertArchitecture.Roberta,
        null))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumn));