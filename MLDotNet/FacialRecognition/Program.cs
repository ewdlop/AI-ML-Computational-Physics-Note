using Microsoft.Azure.CognitiveServices.Vision.Face;
using Microsoft.Azure.CognitiveServices.Vision.Face.Models;

const string SUBSCRIPTION_KEY = "PASTE_YOUR_FACE_SUBSCRIPTION_KEY_HERE";
const string ENDPOINT = "PASTE_YOUR_FACE_ENDPOINT_HERE";
const string RECOGNITION_MODEL4 = RecognitionModel.Recognition04;

string personGroupId = Guid.NewGuid().ToString();
IFaceClient client = new FaceClient(new ApiKeyServiceClientCredentials(SUBSCRIPTION_KEY))
{
    Endpoint = ENDPOINT
};

static async Task IdentifyInPersonGroup(IFaceClient client, string url, string recognitionModel, string personGroupId, CancellationToken cancellationToken)
{
    Console.WriteLine("========IDENTIFY FACES========");
    Console.WriteLine();

    Dictionary<string, string[]> personDictionary = new Dictionary<string, string[]>
    {
        { "Family1-Dad", new[] { "Family1-Dad1.jpg", "Family1-Dad2.jpg" } },
        { "Family1-Mom", new[] { "Family1-Mom1.jpg", "Family1-Mom2.jpg" } },
        { "Family1-Son", new[] { "Family1-Son1.jpg", "Family1-Son2.jpg" } },
        { "Family1-Daughter", new[] { "Family1-Daughter1.jpg", "Family1-Daughter2.jpg" } },
        { "Family2-Lady", new[] { "Family2-Lady1.jpg", "Family2-Lady2.jpg" } },
        { "Family2-Man", new[] { "Family2-Man1.jpg", "Family2-Man2.jpg" } }
    };

    string sourceImageFileName = "identification1.jpg";
    Console.WriteLine($"Create a person group ({personGroupId}).");
    await client.PersonGroup.CreateAsync(
        personGroupId,
        personGroupId,
        recognitionModel,
        cancellationToken: cancellationToken);
    foreach (string groupedFace in personDictionary.Keys)
    {
        Person person = await client.PersonGroupPerson.CreateAsync(
            personGroupId: personGroupId,
            name: groupedFace,
            cancellationToken: cancellationToken);
        Console.WriteLine($"Create a person group person '{groupedFace}'.");

        foreach (string? similarImage in personDictionary[groupedFace])
        {
            Console.WriteLine($"Check whether image is of sufficient quality for recognition");
            IList<DetectedFace> detectedFaces = await client.Face.DetectWithUrlAsync(
                $"{url}{similarImage}",
                recognitionModel: recognitionModel,
                detectionModel: DetectionModel.Detection03,
                returnFaceAttributes: new List<FaceAttributeType> { FaceAttributeType.QualityForRecognition },
                cancellationToken: cancellationToken);
            bool sufficientQuality = true;
            foreach(DetectedFace face in detectedFaces)
            {
                QualityForRecognition? faceQualityForRecognition = face.FaceAttributes.QualityForRecognition;
                //  Only "high" quality images are recommended for person enrollment
                if (faceQualityForRecognition is not null && (faceQualityForRecognition.Value != QualityForRecognition.High))
                {
                    sufficientQuality = false;
                    break;
                }
            }

            if (!sufficientQuality)
            {
                continue;
            }
        }
    }

    Console.WriteLine();
    Console.WriteLine($"Train person group {personGroupId}.");
    await client.PersonGroup.TrainAsync(
        personGroupId,
        cancellationToken: cancellationToken);

    // Wait until the training is completed.
    while (true)
    {
        await Task.Delay(1000, cancellationToken);
        TrainingStatus? trainingStatus = await client.PersonGroup.GetTrainingStatusAsync(personGroupId, cancellationToken: cancellationToken);
        Console.WriteLine($"Training status: {trainingStatus.Status}.");
        if (trainingStatus.Status == TrainingStatusType.Succeeded) { break; }
    }
    Console.WriteLine();

    List<Guid> sourceFaceIds = new List<Guid>();
    // Detect faces from source image url.
    List<DetectedFace> detectedTestFaces = await DetectFaceRecognize(client, $"{url}{sourceImageFileName}", recognitionModel);

    // Add detected faceId to sourceFaceIds.
    foreach (DetectedFace detectedFace in detectedTestFaces) 
    { 
        if(detectedFace.FaceId is not null)sourceFaceIds.Add(detectedFace.FaceId.Value); 
    }

    // Identify the faces in a person group. 
    IList<IdentifyResult> identifyResults = await client.Face.IdentifyAsync(sourceFaceIds, personGroupId, cancellationToken);

    foreach (var identifyResult in identifyResults)
    {
        if (identifyResult.Candidates.Count == 0)
        {
            Console.WriteLine($"No person is identified for the face in: {sourceImageFileName} - {identifyResult.FaceId},");
            continue;
        }
        Person person = await client.PersonGroupPerson.GetAsync(
            personGroupId,
            identifyResult.Candidates[0].PersonId,
            cancellationToken: cancellationToken);
        Console.WriteLine($"Person '{person.Name}' is identified for the face in: {sourceImageFileName} - {identifyResult.FaceId}," +
            $" confidence: {identifyResult.Candidates[0].Confidence}.");
    }
    Console.WriteLine();
}

static async Task<List<DetectedFace>> DetectFaceRecognize(IFaceClient faceClient, string url, string recognition_model)
{
    // Detect faces from image URL. Since only recognizing, use the recognition model 1.
    // We use detection model 3 because we are not retrieving attributes.
    IList<DetectedFace> detectedFaces = await faceClient.Face.DetectWithUrlAsync(
        url,
        recognitionModel: recognition_model,
        detectionModel: DetectionModel.Detection03,
        returnFaceAttributes: new List<FaceAttributeType> { FaceAttributeType.QualityForRecognition });
    List<DetectedFace> sufficientQualityFaces = new List<DetectedFace>();
    foreach (DetectedFace detectedFace in detectedFaces)
    {
        var faceQualityForRecognition = detectedFace.FaceAttributes.QualityForRecognition;
        if (faceQualityForRecognition is not null && (faceQualityForRecognition.Value >= QualityForRecognition.Medium))
        {
            sufficientQualityFaces.Add(detectedFace);
        }
    }
    Console.WriteLine($"{detectedFaces.Count} face(s) with {sufficientQualityFaces.Count} having sufficient quality for recognition detected from image '{Path.GetFileName(url)}'");

    return sufficientQualityFaces.ToList();
}